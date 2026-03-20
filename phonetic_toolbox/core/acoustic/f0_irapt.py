from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import firwin, lfilter, resample_poly


@dataclass
class _CorrParam:
    fft_order: int
    fft_freq_line_size: int
    interp_factor: int
    interp_filter_h_size: int
    interp_filter: np.ndarray
    left_index_actual: int
    right_index_actual: int
    left_index: int
    right_index: int
    actual_indices: np.ndarray
    actual_freqs: np.ndarray
    actual_freqs_num: int
    window: np.ndarray


@dataclass
class _Cfg:
    fs: int
    fs_f0_target: int
    src_sub_ratio: int
    fs_f0: int
    fd: float
    step_sec: float
    step_sub_smp: int
    step_smp: int
    frame_sec: float
    frame_sub_smp: int
    frame_smp: int
    chunk_f0_sec: float
    chunk_f0_size: int
    chunk_f0_freqs: np.ndarray
    f0_limits: Tuple[float, float]
    f0_max_step: int
    f0_final_deviation: float
    max_harmonic_refine: int
    freq_tolerance_refine: float
    initial_f0_value: float
    corr_param: _CorrParam


def _my_fit(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if x1.size == 0 or y1.size == 0 or x2.size == 0:
        return np.zeros(x2.size, dtype=np.float64)
    return np.interp(
        x2.astype(np.float64),
        x1.astype(np.float64),
        y1.astype(np.float64),
        left=float(y1[0]),
        right=float(y1[-1]),
    )


def _my_unwrap_pair(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    phi_b_adj = (
        phi_b
        - np.round((phi_b - phi_a) / (2.0 * np.pi)) * (2.0 * np.pi)
    )
    return np.stack([phi_a, phi_b_adj], axis=1)


def _filter_no_offset(
    imp: np.ndarray,
    frame: np.ndarray,
    h_size: int,
    ground: float,
) -> np.ndarray:
    frame_ex = np.concatenate(
        [frame.astype(np.float64), np.full(h_size, ground, dtype=np.float64)],
    )
    out = lfilter(imp.astype(np.float64), [1.0], frame_ex)
    return out[h_size:]


def _f0_bank_get_samples(
    x: np.ndarray,
    e_mx: np.ndarray,
    m_channels: int,
    s_len: int,
    zi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if s_len > 0:
        zi = np.concatenate([m_channels * x[:s_len][::-1], zi[:-s_len]])
    d_mx = np.reshape(zi, (m_channels, -1), order="F")
    p = np.sum(d_mx * e_mx, axis=1)
    y = np.fft.ifft(p)
    return y, zi


def _get_harmonic_params(
    x: np.ndarray,
    e_mx: np.ndarray,
    m_channels: int,
    s_len: int,
    zi: np.ndarray,
    ind_array0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y1 = np.zeros(m_channels, dtype=np.complex128)
    if s_len > 1:
        y1, zi = _f0_bank_get_samples(
            x[:s_len - 1],
            e_mx,
            m_channels,
            s_len - 1,
            zi,
        )
    y2, zi = _f0_bank_get_samples(
        np.asarray([x[s_len - 1]], dtype=np.float64),
        e_mx,
        m_channels,
        1,
        zi,
    )
    y1_sum = np.sum(y1[ind_array0], axis=0)
    y2_sum = np.sum(y2[ind_array0], axis=0)
    amps = np.abs(y2_sum) * 2.0
    phs = np.angle(y2_sum)
    pair = _my_unwrap_pair(np.angle(y1_sum), phs)
    frc = np.diff(pair, axis=1).ravel()
    return amps.real, frc.real, phs.real, zi


def _take_hparams_whole_sig(
    sig: np.ndarray,
    cfg: _Cfg,
    sustain: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m_channels = 360
    k_merge = 4
    s_len = cfg.step_sub_smp
    if sustain:
        h = firwin(721, 1.0 / m_channels, window="hamming", scale=False)
    else:
        h = firwin(317, 1.0 / m_channels, window="hamming", scale=False)
        pad = m_channels // 2 + 22
        h = np.pad(h, (pad, pad))
    n_h = int(h.size)
    zi = np.zeros(n_h, dtype=np.float64)
    rem = n_h % m_channels
    if rem != 0:
        h = np.pad(h, (0, m_channels - rem))
        zi = np.zeros(h.size, dtype=np.float64)
    e_mx = np.reshape(h, (m_channels, -1), order="F")
    n_frames = int(np.ceil(sig.size / float(s_len)))
    sig_pad = np.pad(
        sig.astype(np.float64),
        (0, int(np.ceil((n_h - 1) / 2.0))),
    )
    n_bins = (m_channels // 2) - (k_merge - 1)
    amps = np.zeros((n_frames, n_bins), dtype=np.float64)
    frcs = np.zeros((n_frames, n_bins), dtype=np.float64)
    phss = np.zeros((n_frames, n_bins), dtype=np.float64)
    offset = int((n_h - 1) / 2 + 1 - s_len)
    warm = sig_pad[:offset]
    _, zi = _f0_bank_get_samples(
        warm.astype(np.float64),
        e_mx,
        m_channels,
        warm.size,
        zi,
    )
    part1 = np.tile(np.arange(1, k_merge + 1)[:, None], (1, n_bins))
    part2 = np.tile(np.arange(0, n_bins)[None, :], (k_merge, 1))
    ind_array0 = (part1 + part2 - 1).astype(np.int64)
    for n in range(1, n_frames + 1):
        st = (n - 1) * s_len + offset
        ed = n * s_len + offset
        frame = sig_pad[st:ed]
        a, f, p, zi = _get_harmonic_params(
            frame,
            e_mx,
            m_channels,
            s_len,
            zi,
            ind_array0,
        )
        amps[n - 1] = a
        frcs[n - 1] = f
        phss[n - 1] = p
    frcs = frcs / (2.0 * np.pi) * float(cfg.fs_f0)
    return amps, frcs, phss


def _value_func_corr_line_fft_interp(
    amp: np.ndarray,
    frc: np.ndarray,
    cfg: _Cfg,
) -> np.ndarray:
    n_params, n_frames = frc.shape
    n_point = cfg.corr_param.actual_freqs.size
    values = np.zeros((n_point, n_frames), dtype=np.float64)
    amp_m = amp.copy()
    frc_m = frc.copy()
    mask_bad = (frc_m < 0.0) | (frc_m > cfg.fs_f0 / 2.0)
    amp_m[mask_bad] = 0.0
    frc_m[mask_bad] = 0.0
    for m in range(n_frames):
        amp_vec = amp_m[:, m]
        frc_vec = frc_m[:, m]
        if np.count_nonzero(amp_vec > 0.0) <= 2:
            continue
        fft_inds = np.rint(
            frc_vec / (cfg.fs_f0 / cfg.corr_param.fft_order) + 1.0,
        ).astype(np.int64)
        fft_amps = np.zeros(
            cfg.corr_param.fft_freq_line_size,
            dtype=np.float64,
        )
        valid = (fft_inds >= 1) & (fft_inds <= fft_amps.size)
        fft_amps[fft_inds[valid] - 1] = amp_vec[valid] ** 2
        fft_amps_full = np.concatenate([fft_amps, fft_amps[-2:0:-1]])
        corr_sig = (
            np.fft.ifft(fft_amps_full).real
            * cfg.corr_param.fft_order / 2.0
        )
        corr_sig = corr_sig[
            cfg.corr_param.left_index - 1:cfg.corr_param.right_index
        ]
        interp_len = (
            cfg.corr_param.right_index
            - cfg.corr_param.left_index + 1
        )
        corr_i = np.zeros(
            interp_len * cfg.corr_param.interp_factor,
            dtype=np.float64,
        )
        corr_i[::cfg.corr_param.interp_factor] = (
            corr_sig * cfg.corr_param.interp_factor
        )
        corr_i = _filter_no_offset(
            cfg.corr_param.interp_filter,
            corr_i,
            cfg.corr_param.interp_filter_h_size,
            0.0,
        )
        values[:, m] = corr_i[cfg.corr_param.actual_indices - 1]
    out = -values.T * cfg.corr_param.window[None, :]
    return out


def _dp_pitch_step(
    value_vec: np.ndarray,
    d_vec: np.ndarray,
    phi: np.ndarray,
    max_step: int,
    leakage: float,
) -> Tuple[int, np.ndarray, np.ndarray]:
    d_vec = d_vec * leakage
    r, c = phi.shape
    phi[:r - 1, :] = phi[1:, :]
    d_new = d_vec.copy()
    d_new[max_step:max_step + c] = value_vec
    idx_arr = np.arange(c)[None, :] + np.arange(0, 2 * max_step + 1)[:, None]
    local = d_vec[idx_arr]
    min_ind = np.argmin(local, axis=0)
    min_val = local[min_ind, np.arange(c)]
    d_new[max_step:max_step + c] += min_val
    phi[r - 1, :] = min_ind + 1
    j = int(np.argmin(d_new[max_step:max_step + c])) + 1
    for rr in range(r - 1, 0, -1):
        tb = int(phi[rr, j - 1])
        j = j - (max_step + 1) + tb
    return j, d_new, phi


def _dp_pitch_full(
    value_mx: np.ndarray,
    max_step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    r, c = value_mx.shape
    d = np.full((r + 1, c + 2 * max_step), np.inf, dtype=np.float64)
    d[0, :] = 0.0
    d[1:, max_step:max_step + c] = value_mx
    phi = np.zeros((r, c), dtype=np.int64)
    for i in range(r):
        for j in range(max_step, max_step + c):
            seg = d[i, j - max_step:j + max_step + 1]
            tb = int(np.argmin(seg))
            d[i + 1, j] += seg[tb]
            phi[i, j - max_step] = tb + 1
    i = r
    j = int(np.argmin(d[r, max_step:max_step + c])) + 1
    q = [j]
    while i > 1 and j >= 1:
        tb = int(phi[i - 1, j - 1])
        i -= 1
        j = j - (max_step + 1) + tb
        if j == 0:
            break
        q.insert(0, j)
    return np.asarray(q, dtype=np.int64), d[r]


def _get_final_f0(
    amp_vec: np.ndarray,
    frc_vec: np.ndarray,
    f0_crude: float,
    cfg: _Cfg,
) -> float:
    if f0_crude <= 0.0:
        return 0.0
    frac = frc_vec / f0_crude
    h_num = np.rint(frac)
    frac = np.abs(frac - h_num)
    good = (frac <= cfg.f0_final_deviation) & (h_num > 0.0)
    if not np.any(good):
        return float(f0_crude)
    amp_g = amp_vec[good]
    frc_g = frc_vec[good]
    h_g = h_num[good]
    den = float(np.sum(amp_g))
    if den <= 0.0:
        return float(f0_crude)
    return float(np.sum(amp_g * (frc_g / h_g)) / den)


def _build_cfg(
    signal_len: int,
    sig_type: Literal["speech", "sustain phonation"],
    min_f0: float,
    max_f0: float,
) -> _Cfg:
    fs = 44100
    fs_f0_target = 6000
    src_sub_ratio = int(round(fs / float(fs_f0_target)))
    fs_f0 = int(round(fs / float(src_sub_ratio)))
    fd = 35.0
    step_sec = 0.005
    step_sub_smp = int(round(step_sec * fs_f0))
    step_smp = step_sub_smp * src_sub_ratio
    frame_sec = 0.05
    frame_sub_smp = int(round(frame_sec * fs_f0 / 2.0) * 2 + 1)
    frame_smp = int(round(frame_sec * fs / 2.0) * 2 + 1)
    if sig_type == "speech":
        chunk_f0_sec = 0.3
        f0_limits = (max(50.0, min_f0), min(450.0, max_f0))
        f0_max_step = 23
    else:
        chunk_f0_sec = signal_len / float(fs)
        f0_limits = (max(65.0, min_f0), min(450.0, max_f0))
        f0_max_step = 2
    chunk_f0_size = int(round(chunk_f0_sec / step_sec))
    chunk_f0_freqs = np.arange(
        int(np.floor(f0_limits[0])),
        int(np.ceil(f0_limits[1])) + 1,
        dtype=np.float64,
    )
    fft_order = 4096 * 4
    fft_freq_line_size = fft_order // 2 + 1
    interp_factor = 2
    interp_h = interp_factor * 12
    interp_filter = firwin(interp_h * 2 + 1, 1.0 / interp_factor)
    left_actual = int(np.floor(fs_f0 / f0_limits[1]) + 1)
    right_actual = int(np.ceil(fs_f0 / f0_limits[0]) + 1)
    sinc_offset = int(np.ceil(interp_h / float(interp_factor)))
    left_index = left_actual - sinc_offset
    right_index = right_actual + sinc_offset
    actual_indices = np.arange(
        sinc_offset * interp_factor + 1,
        (right_actual - left_index) * interp_factor + 2,
        dtype=np.int64,
    )
    n_act = int((right_actual - left_actual) * interp_factor + 1)
    den = np.linspace(
        left_actual - 1,
        right_actual - 1,
        n_act,
        dtype=np.float64,
    )
    actual_freqs = fs_f0 / den
    n_point = int(np.round(np.diff(f0_limits)[0])) + 1
    window = (
        np.arange(n_point, dtype=np.float64) / max(1, n_point - 1)
    ) * 0.25 + 0.75
    x1 = np.arange(f0_limits[0], f0_limits[1] + 1, dtype=np.float64)
    window = _my_fit(x1, window, actual_freqs[::-1])[::-1]
    corr = _CorrParam(
        fft_order=fft_order,
        fft_freq_line_size=fft_freq_line_size,
        interp_factor=interp_factor,
        interp_filter_h_size=interp_h,
        interp_filter=interp_filter,
        left_index_actual=left_actual,
        right_index_actual=right_actual,
        left_index=left_index,
        right_index=right_index,
        actual_indices=actual_indices,
        actual_freqs=actual_freqs,
        actual_freqs_num=actual_freqs.size,
        window=window,
    )
    return _Cfg(
        fs=fs,
        fs_f0_target=fs_f0_target,
        src_sub_ratio=src_sub_ratio,
        fs_f0=fs_f0,
        fd=fd,
        step_sec=step_sec,
        step_sub_smp=step_sub_smp,
        step_smp=step_smp,
        frame_sec=frame_sec,
        frame_sub_smp=frame_sub_smp,
        frame_smp=frame_smp,
        chunk_f0_sec=chunk_f0_sec,
        chunk_f0_size=chunk_f0_size,
        chunk_f0_freqs=chunk_f0_freqs,
        f0_limits=f0_limits,
        f0_max_step=f0_max_step,
        f0_final_deviation=0.1,
        max_harmonic_refine=8,
        freq_tolerance_refine=fd,
        initial_f0_value=0.01,
        corr_param=corr,
    )


def _get_vu_dp(vu_measures: np.ndarray, min_v: int, min_u: int) -> np.ndarray:
    n_frames = vu_measures.size
    d = np.hstack(
        [
            np.tile(vu_measures[:, None], (1, min_v)),
            np.tile((1 - vu_measures)[:, None], (1, min_u)),
        ],
    ).astype(np.float64)
    p = np.zeros((n_frames, min_v + min_u), dtype=np.int64)
    for n in range(1, n_frames):
        for m in range(min_v + min_u):
            if m == 0:
                inds = [min_v + min_u - 1]
            elif m == min_v - 1 or m == min_v + min_u - 1:
                inds = [m - 1, m]
            else:
                inds = [m - 1]
            vals = d[n - 1, inds]
            i_best = int(np.argmax(vals))
            d[n, m] += vals[i_best]
            p[n, m] = inds[i_best]
    pos = np.zeros(n_frames, dtype=np.int64)
    pos[-1] = int(np.argmax(d[-1]))
    for n in range(n_frames - 1, 0, -1):
        pos[n - 1] = p[n, pos[n]]
    signs = np.zeros(n_frames, dtype=np.float64)
    signs[pos < min_v] = 1.0
    return signs


def _irapt1_or_sus(
    signal: np.ndarray,
    cfg: _Cfg,
    sustain: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    sig_f0 = resample_poly(signal, cfg.fs_f0, cfg.fs)
    if sustain:
        b = np.array([1.0, -0.95], dtype=np.float64)
        sig_f0 = lfilter(b, [1.0], sig_f0)
    amp_bank, frc_bank, _ = _take_hparams_whole_sig(
        sig_f0,
        cfg,
        sustain=sustain,
    )
    value_mx = _value_func_corr_line_fft_interp(amp_bank.T, frc_bank.T, cfg)
    if value_mx.size == 0:
        return np.array([]), np.array([])
    if sustain:
        q, _ = _dp_pitch_full(value_mx, cfg.f0_max_step)
        n = min(value_mx.shape[0], q.size)
        f0 = np.zeros(n, dtype=np.float64)
        voc = np.zeros(n, dtype=np.float64)
        for i in range(n):
            idx = max(0, min(cfg.corr_param.actual_freqs_num - 1, q[i] - 1))
            f0_c = float(cfg.corr_param.actual_freqs[idx])
            f0[i] = _get_final_f0(amp_bank[i], frc_bank[i], f0_c, cfg)
            voc[i] = -float(np.min(value_mx[i]))
        vu = _get_vu_dp((voc > 0.002).astype(np.float64), 10, 5)
        f0 = np.where(vu > 0.0, f0, 0.0)
        return f0, voc
    chunk_n = max(1, cfg.chunk_f0_size)
    s_val = max(1, value_mx.shape[0] - chunk_n + 1)
    chunk0 = value_mx[:chunk_n]
    q0, d_vec = _dp_pitch_full(chunk0, cfg.f0_max_step)
    phi = np.zeros((chunk_n, cfg.corr_param.actual_freqs_num), dtype=np.int64)
    f0 = np.zeros(s_val, dtype=np.float64)
    voc = np.zeros(s_val, dtype=np.float64)
    if q0.size > 0:
        f0[0] = cfg.corr_param.actual_freqs[q0[0] - 1]
    en = np.sum(amp_bank[:chunk_n] ** 2, axis=1)
    en = np.maximum(en, 1e-4)
    if chunk_n <= voc.size:
        voc[:chunk_n] = -np.min(chunk0, axis=1) / en
    for n in range(1, s_val):
        vv = value_mx[chunk_n + n - 1]
        qn, d_vec, phi = _dp_pitch_step(vv, d_vec, phi, cfg.f0_max_step, 0.95)
        qn = max(1, min(cfg.corr_param.actual_freqs_num, qn))
        f0_c = float(cfg.corr_param.actual_freqs[qn - 1])
        f0[n] = _get_final_f0(amp_bank[n], frc_bank[n], f0_c, cfg)
        en_n = max(1e-4, float(np.sum(amp_bank[chunk_n + n - 1] ** 2)))
        voc[n] = -float(np.min(vv)) / en_n
    return f0, voc


def _sinc_hash_table() -> np.ndarray:
    try:
        mat_path = Path(__file__).with_name("Sinc_hash_1000.mat")
        mat = loadmat(str(mat_path))
        table = np.asarray(mat.get("Sinc_hash"), dtype=np.float64)
        if table.ndim == 2 and table.shape[1] == 200:
            return table
    except Exception:
        pass
    frac = np.linspace(0.0, 1.0, 1001, dtype=np.float64)[:, None]
    n = np.arange(-99, 101, dtype=np.float64)[None, :]
    table = np.sinc(n - frac)
    win = np.hamming(200)[None, :]
    table = table * win
    row_sum = np.sum(table, axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return table / row_sum


def _sample_with_sinc_hash(
    sig: np.ndarray,
    center_idx: int,
    frac: float,
    sinc_hash: np.ndarray,
) -> float:
    half = 100
    left = center_idx - half
    right = left + 201
    lz = 0 if left >= 0 else -left
    rz = 0 if right < sig.size else right - sig.size + 1
    st = max(0, left)
    ed = min(sig.size - 1, right)
    frame = sig[st:ed + 1]
    if lz > 0:
        frame = np.concatenate([np.zeros(lz, dtype=np.float64), frame])
    if rz > 0:
        frame = np.concatenate([frame, np.zeros(rz, dtype=np.float64)])
    if frame.size < 202:
        frame = np.pad(frame, (0, 202 - frame.size))
    elif frame.size > 202:
        frame = frame[:202]
    frac_idx = int(np.clip(np.rint(frac * 1000.0), 0, 1000))
    return float(np.sum(sinc_hash[frac_idx] * frame[1:-1]))


def _phase_warp_signal(
    sig: np.ndarray,
    f0: np.ndarray,
    step_sub_smp: int,
    fs_f0: int,
    phase_stages: int = 40,
) -> np.ndarray:
    if sig.size == 0 or f0.size == 0:
        return np.array([])
    idx_marks = np.arange(0, f0.size, dtype=np.float64) * step_sub_smp
    sample_idx = np.arange(sig.size, dtype=np.float64)
    frc = _my_fit(idx_marks, f0, sample_idx)
    phase_step = 2.0 * np.pi / float(phase_stages)
    sinc_hash = _sinc_hash_table()
    ph_cum = 0.0
    out: list[float] = []
    for n in range(sig.size):
        ph_last = ph_cum
        ph_cum += float(frc[n]) / float(fs_f0) * 2.0 * np.pi
        p_points = int(np.floor(ph_cum / phase_step))
        if p_points <= 0:
            continue
        targets = np.arange(1, p_points + 1, dtype=np.float64) * phase_step
        moments = _my_fit(
            np.asarray([ph_last, ph_cum], dtype=np.float64),
            np.asarray([0.0, 1.0], dtype=np.float64),
            targets,
        )
        for m in moments:
            out.append(_sample_with_sinc_hash(sig, n, float(m), sinc_hash))
        ph_cum -= p_points * phase_step
    return np.asarray(out, dtype=np.float64)


def _harmonic_params_from_warped(
    warped_sig: np.ndarray,
    f0: np.ndarray,
    fs_f0: int,
    max_harmonic_refine: int,
    freq_tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_frames = f0.size
    n_h = max_harmonic_refine
    if warped_sig.size < 64 or n_frames == 0:
        return np.zeros((n_h, n_frames)), np.zeros((n_h, n_frames))
    centers = np.rint(
        np.linspace(0, warped_sig.size - 1, n_frames, dtype=np.float64),
    ).astype(np.int64)
    win_len = 1024
    half = win_len // 2
    window = np.hamming(win_len)
    amps = np.zeros((n_h, n_frames), dtype=np.float64)
    frcs = np.zeros((n_h, n_frames), dtype=np.float64)
    for i, c in enumerate(centers):
        st = c - half
        ed = st + win_len
        lz = max(0, -st)
        rz = max(0, ed - warped_sig.size)
        st = max(0, st)
        ed = min(warped_sig.size, ed)
        frame = warped_sig[st:ed]
        if lz > 0:
            frame = np.concatenate([np.zeros(lz, dtype=np.float64), frame])
        if rz > 0:
            frame = np.concatenate([frame, np.zeros(rz, dtype=np.float64)])
        if frame.size != win_len:
            frame = np.pad(frame, (0, max(0, win_len - frame.size)))[:win_len]
        spec = np.abs(np.fft.rfft(frame * window))
        freqs = np.fft.rfftfreq(win_len, d=1.0 / float(fs_f0))
        f0_i = max(float(f0[i]), 1e-6)
        for h in range(1, n_h + 1):
            target = f0_i * (h + 1)
            low = max(0.0, target - freq_tolerance)
            high = min(fs_f0 / 2.0, target + freq_tolerance)
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            if idx.size == 0:
                continue
            k = idx[int(np.argmax(spec[idx]))]
            amps[h - 1, i] = float(spec[k])
            frcs[h - 1, i] = float(freqs[k])
    return amps, frcs


def _irapt2_refine(
    signal: np.ndarray,
    cfg: _Cfg,
    f0: np.ndarray,
) -> np.ndarray:
    if f0.size == 0:
        return f0
    sig_f0 = resample_poly(signal, cfg.fs_f0, cfg.fs).astype(np.float64)
    warped_sig = _phase_warp_signal(
        sig_f0,
        f0,
        cfg.step_sub_smp,
        cfg.fs_f0,
        phase_stages=40,
    )
    if warped_sig.size == 0:
        return f0
    amp, frc = _harmonic_params_from_warped(
        warped_sig,
        f0,
        cfg.fs_f0,
        cfg.max_harmonic_refine,
        cfg.freq_tolerance_refine,
    )
    n_samples = f0.size
    amp2 = np.vstack(
        [
            np.full((1, n_samples), cfg.initial_f0_value, dtype=np.float64),
            amp,
        ],
    )
    dev = np.tile(
        np.asarray(
            [1.0] + list(range(1, cfg.max_harmonic_refine + 1)),
            dtype=np.float64,
        ),
        (n_samples, 1),
    )
    frc2 = np.vstack([f0[None, :], frc]).T / dev
    f0_rep = np.tile(f0[:, None], (1, cfg.max_harmonic_refine + 1))
    valid = (
        np.abs(frc2 - f0_rep) < cfg.freq_tolerance_refine
    ) & (frc2 > 0.0)
    num = np.sum(valid * amp2.T * frc2, axis=1)
    den = np.sum(valid * amp2.T, axis=1)
    den[den == 0.0] = 1.0
    f0_ref = num / den
    f0_ref = np.where(np.isfinite(f0_ref) & (f0_ref > 0.0), f0_ref, f0)
    pad = np.pad(f0_ref, (1, 1), mode="edge")
    return (
        0.25 * pad[:-2]
        + 0.5 * pad[1:-1]
        + 0.25 * pad[2:]
    )


def irapt(
    sig: np.ndarray,
    fs: int,
    est_type: Literal["irapt1", "irapt2"] = "irapt1",
    sig_type: Literal["speech", "sustain phonation"] = "sustain phonation",
    min_f0: float = 50.0,
    max_f0: float = 450.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sig.size == 0 or fs <= 0:
        return np.array([]), np.array([]), np.array([])
    x = sig.astype(np.float64, copy=False)
    if fs != 44100:
        x = resample_poly(x, 44100, fs)
    cfg = _build_cfg(x.size, sig_type, min_f0, max_f0)
    sustain = sig_type != "speech"
    f0, voc = _irapt1_or_sus(x, cfg, sustain=sustain)
    if est_type == "irapt2":
        f0 = _irapt2_refine(x, cfg, f0)
    time_marks = np.arange(f0.size, dtype=np.float64) * cfg.step_smp / cfg.fs
    return f0, voc, time_marks
