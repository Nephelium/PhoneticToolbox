import importlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from phonetic_toolbox.core.acoustic.f0_irapt import irapt

parselmouth: Any = None
try:
    parselmouth = importlib.import_module("parselmouth")
except Exception:
    pass


def _extract_f0_for_wm(
    y: np.ndarray,
    fs: int,
    min_f0: float,
    max_f0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        f0_irapt, _, marks_irapt = irapt(
            y,
            fs,
            est_type="irapt1",
            sig_type="sustain phonation",
            min_f0=min_f0,
            max_f0=max_f0,
        )
        valid_irapt = np.isfinite(f0_irapt) & (f0_irapt > 0.0) & np.isfinite(
            marks_irapt,
        )
        if np.count_nonzero(valid_irapt) >= 2:
            return f0_irapt[valid_irapt], marks_irapt[valid_irapt]
    except Exception:
        pass
    if parselmouth is None or y.size == 0:
        return np.array([]), np.array([])
    snd = parselmouth.Sound(
        y.astype(np.float64),
        sampling_frequency=float(fs),
    )
    safe_min_f0 = max(1.0, float(min_f0))
    pitch_obj: Any = None
    try:
        pitch_obj = snd.to_pitch(
            time_step=None,
            pitch_floor=safe_min_f0,
            pitch_ceiling=float(max_f0),
        )
    except Exception:
        try:
            pitch_obj = snd.to_pitch_ac(
                time_step=None,
                pitch_floor=safe_min_f0,
                pitch_ceiling=float(max_f0),
            )
        except Exception:
            return np.array([]), np.array([])
    f0 = np.asarray(pitch_obj.selected_array["frequency"], dtype=np.float64)
    time_marks = np.asarray(pitch_obj.xs(), dtype=np.float64)
    valid = np.isfinite(f0) & (f0 > 0.0) & np.isfinite(time_marks)
    if np.count_nonzero(valid) < 2:
        return np.array([]), np.array([])
    return f0[valid], time_marks[valid]


def _wm_phase_const(
    sig: np.ndarray,
    f0_hz: np.ndarray,
    time_marks: np.ndarray,
    fs: int,
) -> np.ndarray:
    ln = int(sig.size)
    if ln < 3 or f0_hz.size < 2 or time_marks.size < 2:
        return np.array([], dtype=np.int64)
    time_line: NDArray[np.float64] = np.arange(1, ln + 1, dtype=np.float64)
    sample_marks = np.asarray(time_marks * float(fs), dtype=np.float64)
    sample_marks = np.clip(sample_marks, 1.0, float(ln))
    phase_f0: NDArray[np.float64] = np.asarray(
        np.interp(
            time_line,
            sample_marks,
            f0_hz,
            left=float(f0_hz[0]),
            right=float(f0_hz[-1]),
        ),
        dtype=np.float64,
    )
    phase_f0 = (phase_f0 / float(fs)) * (2.0 * np.pi)
    cur_cum = 0.0
    n = 0
    while cur_cum < (2.0 * np.pi) and n < ln:
        n += 1
        cur_cum += float(phase_f0[n - 1])
    first_period = n - 1
    if first_period <= 0:
        return np.array([], dtype=np.int64)
    periods: list[int] = [first_period]
    t_prev = sig[:first_period]
    cur_period = 1
    cur_cum -= 2.0 * np.pi
    while n < (ln - periods[-1]):
        if cur_cum > (2.0 * np.pi):
            cur_p = cur_period - 1
            offset = int(np.rint(cur_period * 0.10))
            if offset <= 0:
                biases = np.array([0], dtype=np.int64)
            else:
                biases = np.rint(
                    np.arange(-offset * 0.8, offset * 1.2 + 1.0),
                ).astype(np.int64)
                biases = np.unique(biases)
            best_bias = 0
            best_err: Optional[float] = None
            for bias_value in biases.tolist():
                bias = int(bias_value)
                front_edge = (n - 1) + bias
                if bias >= 0:
                    tc = periods[-1]
                else:
                    tc = periods[-1] + bias
                if tc <= 0:
                    continue
                back_edge = front_edge - tc + 1
                if back_edge < 1 or front_edge > ln or back_edge > front_edge:
                    continue
                start = back_edge - 1
                end = front_edge
                t_cur = sig[start:end]
                if t_cur.size != tc or t_prev.size < tc:
                    continue
                err = float(np.mean(np.abs(t_prev[-tc:] - t_cur)))
                if best_err is None or err < best_err:
                    best_err = err
                    best_bias = bias
            n = (n - 1) + best_bias
            t_new = cur_p + best_bias
            if t_new <= 0:
                t_new = max(1, cur_p)
            if n <= 0 or n > ln:
                break
            periods.append(int(t_new))
            start = n - int(t_new)
            if start < 0:
                break
            t_prev = sig[start:n]
            cur_period = 0
            cur_cum = 0.0
        else:
            if n >= ln:
                break
            n += 1
            cur_cum += float(phase_f0[n - 1])
            cur_period += 1
    t0 = np.asarray(periods, dtype=np.int64)
    return t0[t0 > 0]


def _amp_extract(periods: np.ndarray, sig: np.ndarray) -> np.ndarray:
    if periods.size == 0:
        return np.array([])
    amps: NDArray[np.float64] = np.zeros(periods.size, dtype=np.float64)
    start = 0
    valid_count = 0
    for idx, p in enumerate(periods):
        plen = int(p)
        if plen <= 0:
            break
        end = min(start + plen, sig.size)
        if end <= start:
            break
        cycle = sig[start:end]
        amps[idx] = float(np.max(cycle) - np.min(cycle))
        valid_count += 1
        start += plen
        if start >= sig.size:
            break
    return amps[:valid_count]


def get_period_and_amplitude_vectors(
    y: np.ndarray,
    fs: int,
    min_f0: float,
    max_f0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f0_hz, time_marks = _extract_f0_for_wm(y, fs, min_f0, max_f0)
    if f0_hz.size < 2:
        return np.array([]), np.array([]), np.array([])
    periods_samples = _wm_phase_const(y, f0_hz, time_marks, fs)
    if periods_samples.size < 2:
        return np.array([]), np.array([]), np.array([])
    amps = _amp_extract(periods_samples, y)
    n = min(periods_samples.size, amps.size)
    if n < 2:
        return np.array([]), np.array([]), np.array([])
    periods_samples = periods_samples[:n]
    amps = amps[:n]
    pulse_times = np.concatenate(
        ([0.0], np.cumsum(periods_samples.astype(np.float64))),
    ) / float(fs)
    periods_sec = periods_samples.astype(np.float64) / float(fs)
    return pulse_times, periods_sec, amps


def compute_local_perturbation(x: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    mean_val = float(np.mean(x))
    if mean_val == 0.0:
        return np.nan
    return float(np.mean(np.abs(np.diff(x))) / mean_val * 100.0)


def compute_pq(x: np.ndarray, n_points: int) -> float:
    if x.size < n_points:
        return np.nan
    moving_avg = np.convolve(
        x,
        np.ones(n_points, dtype=np.float64) / float(n_points),
        mode="valid",
    )
    offset = (n_points - 1) // 2
    x_segment = x[offset:offset + moving_avg.size]
    mean_val = float(np.mean(x))
    if mean_val == 0.0:
        return np.nan
    return float(np.mean(np.abs(x_segment - moving_avg)) / mean_val * 100.0)


def compute_jitter_shimmer(
    y: np.ndarray,
    fs: int,
    frameshift_ms: float,
    window_ms: int,
    voiced_mask: Optional[np.ndarray] = None,
    min_f0: float = 75.0,
    max_f0: float = 600.0,
) -> Dict[str, np.ndarray]:
    n_frames = int(
        round(y.size / float(fs) * 1000.0 / float(frameshift_ms)),
    )
    keys = [
        "Jitter_Local",
        "Jitter_RAP",
        "Jitter_PPQ5",
        "Shimmer_Local",
        "Shimmer_APQ3",
        "Shimmer_APQ5",
        "Shimmer_APQ11",
    ]
    results = {key: np.full(n_frames, np.nan) for key in keys}
    pulse_times, t_all, a_all = get_period_and_amplitude_vectors(
        y,
        fs,
        min_f0,
        max_f0,
    )
    if t_all.size == 0:
        return results
    window_sec = float(window_ms) / 1000.0
    frame_centers = (
        (np.arange(n_frames, dtype=np.float64) + 0.5)
        * float(frameshift_ms) / 1000.0
    )
    for i, t_center in enumerate(frame_centers):
        if voiced_mask is not None and voiced_mask.size > 0:
            mask_idx = min(i, voiced_mask.size - 1)
            if not bool(voiced_mask[mask_idx]):
                continue
        t_start = t_center - window_sec / 2.0
        t_end = t_center + window_sec / 2.0
        idx_start = int(np.searchsorted(pulse_times, t_start))
        idx_end = int(np.searchsorted(pulse_times, t_end))
        valid_end = min(idx_end, t_all.size)
        if valid_end <= idx_start:
            continue
        t_frame = t_all[idx_start:valid_end]
        a_frame = a_all[idx_start:valid_end]
        if t_frame.size >= 2:
            results["Jitter_Local"][i] = compute_local_perturbation(t_frame)
            results["Shimmer_Local"][i] = compute_local_perturbation(a_frame)
        if t_frame.size >= 3:
            results["Jitter_RAP"][i] = compute_pq(t_frame, 3)
            results["Shimmer_APQ3"][i] = compute_pq(a_frame, 3)
        if t_frame.size >= 5:
            results["Jitter_PPQ5"][i] = compute_pq(t_frame, 5)
            results["Shimmer_APQ5"][i] = compute_pq(a_frame, 5)
        if a_frame.size >= 11:
            results["Shimmer_APQ11"][i] = compute_pq(a_frame, 11)
    return results
