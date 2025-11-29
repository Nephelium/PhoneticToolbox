from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import tempfile
import subprocess
import shutil
import os
import sys
from scipy.io import loadmat, savemat, wavfile
from scipy.optimize import fminbound, minimize

try:
    import parselmouth
except Exception:  # pragma: no cover
    parselmouth = None  # 在未安装时允许界面运行


def compute_praat_f0_formants(
    wav_path: Path,
    frameshift_ms: int,
    min_f0: int,
    max_f0: int,
    method: str = "cc",
) -> Dict[str, Any]:
    """使用 Praat/Parselmouth 计算 F0 与 Burg 形式的 F1–F4。

    返回的键名与 MATLAB 版本保持一致，以便下游 func_buildMData 兼容：
    - pF0: Praat F0
    - pF1..pF4: Praat Formants
    - pB1..pB4: Praat Bandwidths
    - Fs: 采样率
    """

    if parselmouth is None:
        raise RuntimeError("缺少依赖: praat-parselmouth 未安装")

    snd = parselmouth.Sound(str(wav_path))
    fs = int(round(snd.sampling_frequency))
    time_step = frameshift_ms / 1000.0

    try:
        if method and method.lower() == "cc":
            pitch = parselmouth.praat.call(
                snd,
                "To Pitch (cc)",
                time_step,
                float(min_f0),
                float(max_f0),
            )
        else:
            pitch = snd.to_pitch(
                time_step=time_step,
                pitch_floor=float(min_f0),
                pitch_ceiling=float(max_f0),
            )
    except Exception:
        pitch = snd.to_pitch(
            time_step=time_step,
            pitch_floor=float(min_f0),
            pitch_ceiling=float(max_f0),
        )
    pF0_raw: np.ndarray = np.array(pitch.selected_array["frequency"], dtype=float)
    # 将无声段的F0设为NaN，遵循Praat输出：0 表示无声
    pF0_values: List[float] = list(np.where(pF0_raw <= 0.0, np.nan, pF0_raw))

    formant = snd.to_formant_burg(
        time_step=time_step,
        max_number_of_formants=4,
        maximum_formant=6000.0,
        window_length=0.025,
        pre_emphasis_from=50.0,
    )

    # 对齐帧数量
    n = len(pF0_values)
    def series_for_track(track: int) -> List[float]:
        vals = []
        for i in range(n):
            t = i * time_step
            try:
                v = formant.get_value_at_time(formant_number=track, time=t)
                vals.append(v if v is not None and v > 0 else np.nan)
            except Exception:
                vals.append(np.nan)
        return vals

    def bandwidth_for_track(track: int) -> List[float]:
        vals = []
        for i in range(n):
            t = i * time_step
            try:
                v = formant.get_bandwidth_at_time(formant_number=track, time=t)
                vals.append(v if v is not None and v > 0 else np.nan)
            except Exception:
                vals.append(np.nan)
        return vals

    result: Dict[str, Any] = {
        "pF0": np.array(pF0_values, dtype=float),
        "pF1": np.array(series_for_track(1), dtype=float),
        "pF2": np.array(series_for_track(2), dtype=float),
        "pF3": np.array(series_for_track(3), dtype=float),
        "pF4": np.array(series_for_track(4), dtype=float),
        "pB1": np.array(bandwidth_for_track(1), dtype=float),
        "pB2": np.array(bandwidth_for_track(2), dtype=float),
        "pB3": np.array(bandwidth_for_track(3), dtype=float),
        "pB4": np.array(bandwidth_for_track(4), dtype=float),
        "Fs": fs,
    }

    return result


def compute_straight_f0(
    wav_path: Path,
    frameshift_ms: int,
    min_f0: float,
    max_f0: float,
    matlab_bin: str | None = None,
) -> Dict[str, Any]:
    return {"strF0": np.array([], dtype=float), "Fs": 0}


def read_wav_mono_float(wav_path: Path) -> Tuple[int, np.ndarray]:
    fs, y = wavfile.read(str(wav_path))
    if y.ndim > 1:
        y = y[:, 0]
    if np.issubdtype(y.dtype, np.integer):
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
        elif y.dtype == np.uint8:
            y = (y.astype(np.float32) - 128.0) / 128.0
        else:
            m = float(np.max(np.abs(y)))
            y = y.astype(np.float32) / (m + 1e-9)
    else:
        y = y.astype(np.float32)
        m = float(np.max(np.abs(y)))
        if m > 0:
            y = y / m
    return fs, y


def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    try:
        import sys
        try:
            from opensauce.helpers import round_half_away_from_zero as rha
        except Exception:
            base = Path(__file__).resolve().parent
            candidate = base / ".." / ".." / "opensauce-python-master"
            sys.path.append(str(candidate))
            from opensauce.helpers import round_half_away_from_zero as rha
        return rha(x)
    except Exception:
        return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(int)


def compute_shrp_f0(
    wav_path: Path,
    frameshift_ms: int,
    min_f0: float,
    max_f0: float,
    shr_threshold: float = 0.4,
) -> Dict[str, Any]:
    try:
        import sys
        try:
            from opensauce.shrp import shr_pitch
        except Exception:
            base = Path(__file__).resolve().parent
            candidate = base / ".." / ".." / "opensauce-python-master"
            sys.path.append(str(candidate))
            from opensauce.shrp import shr_pitch
        fs, y = read_wav_mono_float(wav_path)
        nframes = int(round(y.size / float(fs) * 1000.0 / float(frameshift_ms)))
        if nframes <= 0:
            return {"shrF0": np.array([], dtype=float), "SHR": np.array([], dtype=float), "Fs": fs}
        shr, f0 = shr_pitch(
            y.astype(float),
            fs,
            window_length=40,
            frame_shift=frameshift_ms,
            min_pitch=float(min_f0),
            max_pitch=float(max_f0),
            shr_threshold=float(shr_threshold),
            frame_precision=2,
            datalen=nframes,
        )
        f0 = np.array(f0, dtype=float)
        shr = np.array(shr, dtype=float)
        if f0.shape[0] > nframes:
            f0 = f0[:nframes]
        if shr.shape[0] > nframes:
            shr = shr[:nframes]
        return {"shrF0": f0, "SHR": shr, "Fs": fs}
    except Exception:
        fs, _ = read_wav_mono_float(wav_path)
        nframes = int(round(Path(wav_path).stat().st_size / max(fs, 1)))
        return {"shrF0": np.array([], dtype=float), "SHR": np.array([], dtype=float), "Fs": fs}


def _parse_reaper_est_f0(path: Path) -> Tuple[List[float], List[int], List[float]]:
    times: List[float] = []
    voiced: List[int] = []
    values: List[float] = []
    try:
        with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
            in_header = True
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if in_header:
                    if s == "EST_Header_End":
                        in_header = False
                    continue
                parts = s.split()
                if len(parts) < 3:
                    continue
                try:
                    t = float(parts[0])
                    v = int(float(parts[1]))
                    val = float(parts[2])
                except Exception:
                    continue
                times.append(t)
                voiced.append(v)
                values.append(val)
    except Exception:
        pass
    return times, voiced, values


def _find_reaper_bin(explicit: str | None = None) -> str:
    def _resolve_candidate(path_like: str | Path) -> str | None:
        try:
            p = Path(path_like)
            if p.is_file():
                return str(p.resolve())
            if p.is_dir():
                exe = p / "reaper.exe"
                if exe.is_file():
                    return str(exe.resolve())
                exe2 = p / "reaper"
                if exe2.is_file():
                    return str(exe2.resolve())
        except Exception:
            return None
        return None

    if explicit:
        r = _resolve_candidate(explicit)
        if r:
            return r
        w = shutil.which(str(explicit))
        if w:
            return w

    for name in ("reaper", "reaper.exe"):
        w = shutil.which(name)
        if w:
            return w

    proj = Path(__file__).resolve().parents[2]
    meipass = Path(getattr(sys, "_MEIPASS", Path.cwd()))
    cands = [
        meipass / "reaper.exe",
        meipass / "reaper" / "reaper.exe",
        Path.cwd() / "reaper.exe",
        Path.cwd() / "reaper",
        proj / "REAPER-master" / "reaper.exe",
        Path(__file__).resolve().parent / ".." / "reaper.exe",
        Path(__file__).resolve().parent / ".." / "reaper",
    ]
    for cand in cands:
        r = _resolve_candidate(cand)
        if r:
            return r
    raise RuntimeError("未找到 reaper 可执行文件")


def compute_reaper_f0(
    wav_path: Path,
    frame_interval_sec: float,
    min_f0: float,
    max_f0: float,
    hilbert: bool = False,
    no_highpass: bool = False,
    reaper_bin: str | None = None,
) -> Dict[str, Any]:
    wav_path = Path(wav_path)
    rb = _find_reaper_bin(reaper_bin)
    import tempfile
    tmpdir = tempfile.TemporaryDirectory()
    f0_out = Path(tmpdir.name) / "out.f0"
    try:
        cmd = [
            rb,
            "-i", str(wav_path),
            "-f", str(f0_out),
            "-a",
            "-e", str(float(frame_interval_sec)),
            "-m", str(float(min_f0)),
            "-x", str(float(max_f0)),
        ]
        if hilbert:
            cmd.append("-t")
        if no_highpass:
            cmd.append("-s")
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode(errors="ignore") or "reaper failed")
        times, voiced, values = _parse_reaper_est_f0(f0_out)
        t_arr = np.array(times, dtype=float)
        v_arr = np.array(voiced, dtype=int)
        val_arr = np.array(values, dtype=float)
        out_series = np.where(val_arr > 0.0, val_arr, np.nan)
        return {"rTimes": t_arr, "rVoiced": v_arr, "rF0Raw": val_arr, "rF0": out_series}
    finally:
        try:
            tmpdir.cleanup()
        except Exception:
            pass


def _segment_for_frame(y: np.ndarray, fs: int, frameshift_ms: int, k: int, N_periods: int, f0_curr: float) -> np.ndarray:
    """为第 k 帧切片 N_periods 个基音周期长度的语音片段。
    遵循 MATLAB VoiceSauce: 样本中心 ks = k * sampleshift, N0 = Fs / F0。
    无声或边界不足返回空数组。
    """
    if np.isnan(f0_curr) or f0_curr <= 0:
        return np.array([], dtype=float)
    sampleshift = int(round(fs / 1000.0 * frameshift_ms))
    ks = int(round(k * sampleshift))
    N0 = fs / float(f0_curr)
    ystart = int(round(ks - N_periods / 2.0 * N0))
    yend = int(round(ks + N_periods / 2.0 * N0)) - 1
    if ystart <= 0 or yend >= len(y):
        return np.array([], dtype=float)
    seg = y[ystart:yend].astype(float)
    return seg


def _harmonic_mag_db(yseg: np.ndarray, fs: int, freq: float) -> float:
    """计算 yseg 在频率 freq 的复指数投影幅度（dB）。
    公式来源：func_EstMaxVal 与 func_GetHarmonics（UCLA SPAPL）。
    """
    if yseg.size == 0 or freq <= 0:
        return np.nan
    n = np.arange(yseg.size, dtype=float)
    v = np.exp(-1j * 2.0 * np.pi * freq * n / float(fs))
    amp = np.abs(np.dot(yseg, v.conjugate()))
    return 20.0 * np.log10(amp + 1e-12)


def _search_peak_mag_db(yseg: np.ndarray, fs: int, f_est: float, df: float, range_frac: float) -> Tuple[float, float]:
    """在 [f_est*(1-range_frac), f_est*(1+range_frac)] 范围内以步长 df 搜索最大幅度（向量化）。
    返回 (max_db, max_freq)。
    """
    if yseg.size == 0 or np.isnan(f_est) or f_est <= 0:
        return (np.nan, np.nan)
    fmin = max(1.0, f_est * (1.0 - range_frac))
    nyq = fs / 2.0 - 1.0
    fmax = min(nyq, f_est * (1.0 + range_frac))
    if fmax <= fmin:
        return (np.nan, np.nan)
    grid = np.arange(fmin, fmax + df, df, dtype=float)
    n = np.arange(yseg.size, dtype=float)
    # 计算所有频率的复指数基底，再与段做乘积求幅度（向量化）
    # exp_matrix 的形状为 (len(grid), len(yseg))
    exp_matrix = np.exp(-1j * 2.0 * np.pi * np.outer(grid, n) / float(fs))
    amps = np.abs(exp_matrix.conj() @ yseg.astype(float))
    mags_db = 20.0 * np.log10(amps + 1e-12)
    if mags_db.size == 0 or np.all(np.isnan(mags_db)):
        return (np.nan, np.nan)
    inx = int(np.nanargmax(mags_db))
    return (float(mags_db[inx]), float(grid[inx]))


def compute_harmonics_H1H2H4(
    y: np.ndarray,
    fs: int,
    frameshift_ms: int,
    F0: np.ndarray,
    N_periods: int,
    voiced_mask: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    nframes = int(F0.shape[0])
    H1 = np.full(nframes, np.nan, dtype=float)
    H2 = np.full(nframes, np.nan, dtype=float)
    H4 = np.full(nframes, np.nan, dtype=float)
    for k in range(nframes):
        if voiced_mask is not None and k < voiced_mask.shape[0] and not bool(voiced_mask[k]):
            continue
        f0c = float(F0[k])
        seg = _segment_for_frame(y, fs, frameshift_ms, k + 1, N_periods, f0c)
        if seg.size == 0 or np.isnan(f0c) or f0c <= 0:
            continue
        def _maximize_by_matlab_style(f_est: float) -> float:
            df_range = float(np.round(f_est * 0.1))
            fmin_b = max(1.0, f_est - df_range)
            fmax_b = min(fs / 2.0 - 1.0, f_est + df_range)
            if fmax_b <= fmin_b:
                return np.nan
            def obj(xx: float) -> float:
                x = float(xx)
                if x < fmin_b or x > fmax_b:
                    # penalty outside bounds
                    return 1e9
                return -_harmonic_mag_db(seg, fs, x)
            try:
                res = minimize(lambda x: obj(x[0]), x0=np.array([f_est], dtype=float), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4, "maxfev": 300})
                xopt = float(np.clip(res.x[0], fmin_b, fmax_b))
                return float(_harmonic_mag_db(seg, fs, xopt))
            except Exception:
                try:
                    # fallback to bounded golden-section search
                    xopt = fminbound(lambda x: -_harmonic_mag_db(seg, fs, float(x)), fmin_b, fmax_b, xtol=1e-4, maxfun=200)
                    return float(_harmonic_mag_db(seg, fs, float(xopt)))
                except Exception:
                    return np.nan
        H1[k] = _maximize_by_matlab_style(f0c)
        H2[k] = _maximize_by_matlab_style(2.0 * f0c)
        H4[k] = _maximize_by_matlab_style(4.0 * f0c)
    return {"H1": H1, "H2": H2, "H4": H4}


def compute_harmonic_at_fixed_freq(
    y: np.ndarray,
    fs: int,
    frameshift_ms: int,
    F0: np.ndarray,
    N_periods: int,
    target_hz: float,
    voiced_mask: np.ndarray | None = None,
) -> np.ndarray:
    """计算固定目标频率附近的谐波幅度（例如 2kHz、5kHz）。
    参考 MATLAB: func_Get2K / func_Get5K。
    """
    nframes = int(F0.shape[0])
    out = np.full(nframes, np.nan, dtype=float)
    for k in range(nframes):
        if voiced_mask is not None and k < voiced_mask.shape[0] and not bool(voiced_mask[k]):
            continue
        f0c = float(F0[k])
        seg = _segment_for_frame(y, fs, frameshift_ms, k + 1, N_periods, f0c)
        if seg.size == 0:
            continue
        if np.isnan(f0c) or f0c <= 0:
            continue
        df = max(1.0, f0c / 20.0)
        range_frac = (1.0 * f0c) / float(target_hz)
        mag, _ = _search_peak_mag_db(seg, fs, target_hz, df=df, range_frac=range_frac)
        out[k] = mag
    return out

def compute_harmonic_at_fixed_freq_with_freq(
    y: np.ndarray,
    fs: int,
    frameshift_ms: int,
    F0: np.ndarray,
    N_periods: int,
    target_hz: float,
    voiced_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    nframes = int(F0.shape[0])
    mags = np.full(nframes, np.nan, dtype=float)
    freqs = np.full(nframes, np.nan, dtype=float)
    for k in range(nframes):
        if voiced_mask is not None and k < voiced_mask.shape[0] and not bool(voiced_mask[k]):
            continue
        f0c = float(F0[k])
        seg = _segment_for_frame(y, fs, frameshift_ms, k + 1, N_periods, f0c)
        if seg.size == 0 or np.isnan(f0c) or f0c <= 0:
            continue
        df = max(1.0, f0c / 20.0)
        range_frac = (1.0 * f0c) / float(target_hz)
        m, f = _search_peak_mag_db(seg, fs, target_hz, df=df, range_frac=range_frac)
        mags[k] = m
        freqs[k] = f
    return mags, freqs

def compute_H42K_corrected(
    H4: np.ndarray,
    H2K: np.ndarray,
    F2K: np.ndarray,
    fs: int,
    F0: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    F3: np.ndarray,
    B1: np.ndarray | None = None,
    B2: np.ndarray | None = None,
    B3: np.ndarray | None = None,
) -> np.ndarray:
    n = int(F0.shape[0])
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        f0 = float(F0[i])
        if np.isnan(f0) or f0 <= 0:
            continue
        h4c = float(H4[i])
        h2kc = float(H2K[i])
        f2k = float(F2K[i]) if i < F2K.shape[0] else np.nan
        f1 = float(F1[i]) if i < F1.shape[0] else np.nan
        f2 = float(F2[i]) if i < F2.shape[0] else np.nan
        f3 = float(F3[i]) if i < F3.shape[0] else np.nan
        b1 = float(B1[i]) if B1 is not None else hawks_miller_bw(f1, f0)
        b2 = float(B2[i]) if B2 is not None else hawks_miller_bw(f2, f0)
        b3 = float(B3[i]) if B3 is not None else hawks_miller_bw(f3, f0)
        if not np.isnan(f1) and not np.isnan(b1):
            h4c -= iseli_correction(4.0 * f0, f1, b1, fs)
        if not np.isnan(f2) and not np.isnan(b2):
            h4c -= iseli_correction(4.0 * f0, f2, b2, fs)
        if not np.isnan(f2k) and not np.isnan(f1) and not np.isnan(b1):
            h2kc -= iseli_correction(f2k, f1, b1, fs)
        if not np.isnan(f2k) and not np.isnan(f2) and not np.isnan(b2):
            h2kc -= iseli_correction(f2k, f2, b2, fs)
        if not np.isnan(f2k) and not np.isnan(f3) and not np.isnan(b3):
            h2kc -= iseli_correction(f2k, f3, b3, fs)
        out[i] = h4c - h2kc
    return out

def compute_2K5K_corrected(
    H2K: np.ndarray,
    F2K: np.ndarray,
    H5K: np.ndarray,
    fs: int,
    F0: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    F3: np.ndarray,
    B1: np.ndarray | None = None,
    B2: np.ndarray | None = None,
    B3: np.ndarray | None = None,
) -> np.ndarray:
    n = int(F0.shape[0])
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        f0 = float(F0[i])
        if np.isnan(f0) or f0 <= 0:
            continue
        h2kc = float(H2K[i])
        h5k = float(H5K[i])
        f2k = float(F2K[i]) if i < F2K.shape[0] else np.nan
        f1 = float(F1[i]) if i < F1.shape[0] else np.nan
        f2 = float(F2[i]) if i < F2.shape[0] else np.nan
        f3 = float(F3[i]) if i < F3.shape[0] else np.nan
        b1 = float(B1[i]) if B1 is not None else hawks_miller_bw(f1, f0)
        b2 = float(B2[i]) if B2 is not None else hawks_miller_bw(f2, f0)
        b3 = float(B3[i]) if B3 is not None else hawks_miller_bw(f3, f0)
        if not np.isnan(f2k) and not np.isnan(f1) and not np.isnan(b1):
            h2kc -= iseli_correction(f2k, f1, b1, fs)
        if not np.isnan(f2k) and not np.isnan(f2) and not np.isnan(b2):
            h2kc -= iseli_correction(f2k, f2, b2, fs)
        if not np.isnan(f2k) and not np.isnan(f3) and not np.isnan(b3):
            h2kc -= iseli_correction(f2k, f3, b3, fs)
        out[i] = h2kc - h5k
    return out


def compute_A1A2A3(
    y: np.ndarray,
    fs: int,
    frameshift_ms: int,
    F0: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    F3: np.ndarray,
    N_periods: int,
    voiced_mask: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    nframes = int(F0.shape[0])
    A1 = np.full(nframes, np.nan, dtype=float)
    A2 = np.full(nframes, np.nan, dtype=float)
    A3 = np.full(nframes, np.nan, dtype=float)
    fftlen = 8192
    fstep = fs / float(fftlen)
    for k in range(nframes):
        if voiced_mask is not None and k < voiced_mask.shape[0] and not bool(voiced_mask[k]):
            continue
        f0c = float(F0[k])
        seg = _segment_for_frame(y, fs, frameshift_ms, k + 1, N_periods, f0c)
        if seg.size == 0:
            continue
        X = np.fft.fft(seg.astype(float), n=fftlen)
        X[X == 0] = 1e-9
        mags = 20.0 * np.log10(np.abs(X[: fftlen // 2]) + 1e-12)
        f1 = float(F1[k]) if k < F1.shape[0] else np.nan
        f2 = float(F2[k]) if k < F2.shape[0] else np.nan
        f3 = float(F3[k]) if k < F3.shape[0] else np.nan
        if not np.isnan(f1):
            lowf = max(0.0, f1 - 0.1 * f1)
            highf = min(fs / 2.0 - fstep, f1 + 0.1 * f1)
            s = int(1 + np.round(lowf / fstep))
            e = int(1 + np.round(highf / fstep))
            s = max(0, min(s, mags.shape[0] - 1))
            e = max(s, min(e, mags.shape[0] - 1))
            sl = mags[s : e + 1]
            if sl.size > 0:
                pos = int(np.argmax(sl))
                A1[k] = float(sl[pos])
        if not np.isnan(f2):
            lowf = max(0.0, f2 - 0.1 * f2)
            highf = min(fs / 2.0 - fstep, f2 + 0.1 * f2)
            s = int(1 + np.round(lowf / fstep))
            e = int(1 + np.round(highf / fstep))
            s = max(0, min(s, mags.shape[0] - 1))
            e = max(s, min(e, mags.shape[0] - 1))
            sl = mags[s : e + 1]
            if sl.size > 0:
                pos = int(np.argmax(sl))
                A2[k] = float(sl[pos])
        if not np.isnan(f3):
            lowf = max(0.0, f3 - 0.1 * f3)
            highf = min(fs / 2.0 - fstep, f3 + 0.1 * f3)
            s = int(1 + np.round(lowf / fstep))
            e = int(1 + np.round(highf / fstep))
            s = max(0, min(s, mags.shape[0] - 1))
            e = max(s, min(e, mags.shape[0] - 1))
            sl = mags[s : e + 1]
            if sl.size > 0:
                pos = int(np.argmax(sl))
                A3[k] = float(sl[pos])
    return {"A1": A1, "A2": A2, "A3": A3}


def iseli_correction(f: float, Fx: float, Bx: float, fs: int) -> float:
    """Iseli & Alwan (1999) 形式的谐波-共振峰校正，返回校正量（dB）。
    对应 MATLAB: func_correct_iseli_z。
    """
    if any(np.isnan(x) for x in [f, Fx, Bx]) or f <= 0 or Fx <= 0 or Bx <= 0:
        return 0.0
    r = np.exp(-np.pi * Bx / float(fs))
    omega_x = 2.0 * np.pi * Fx / float(fs)
    omega = 2.0 * np.pi * f / float(fs)
    a = r ** 2 + 1.0 - 2.0 * r * np.cos(omega_x + omega)
    b = r ** 2 + 1.0 - 2.0 * r * np.cos(omega_x - omega)
    num = r ** 2 + 1.0 - 2.0 * r * np.cos(omega_x)
    return float(-10.0 * (np.log10(a) + np.log10(b)) + 20.0 * np.log10(num))


def hawks_miller_bw(Fx: float, F0: float) -> float:
    """Hawks & Miller (1995) 带宽估计。
    与 MATLAB 的 getbw_HawksMiller 一致。
    """
    if np.isnan(Fx) or Fx <= 0:
        return np.nan
    S = 1.0 + 0.25 * ((F0 if not np.isnan(F0) else 132.0) - 132.0) / 88.0
    C1 = np.array([165.327516, -6.73636734e-1, 1.80874446e-3, -4.52201682e-6, 7.49514000e-9, -4.70219241e-12])
    C2 = np.array([15.8146139, 8.10159009e-2, -9.79728215e-5, 5.28725064e-8, -1.07099364e-11, 7.91528509e-16])
    F = np.array([1.0, Fx, Fx ** 2, Fx ** 3, Fx ** 4, Fx ** 5], dtype=float)
    if Fx < 500.0:
        bw = float(C1 @ F)
    else:
        bw = float(C2 @ F)
    return float(S * bw)


def compute_H1A1A2A3_corrected(
    H1: np.ndarray,
    A1: np.ndarray,
    A2: np.ndarray,
    A3: np.ndarray,
    fs: int,
    F0: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    F3: np.ndarray,
    B1: np.ndarray | None = None,
    B2: np.ndarray | None = None,
    B3: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    n = int(H1.shape[0])
    H1A1c = np.full(n, np.nan, dtype=float)
    H1A2c = np.full(n, np.nan, dtype=float)
    H1A3c = np.full(n, np.nan, dtype=float)
    for i in range(n):
        f0 = float(F0[i])
        f1 = float(F1[i]) if i < F1.shape[0] else np.nan
        f2 = float(F2[i]) if i < F2.shape[0] else np.nan
        f3 = float(F3[i]) if i < F3.shape[0] else np.nan
        if np.isnan(f0) or f0 <= 0:
            continue
        b1 = float(B1[i]) if B1 is not None else hawks_miller_bw(f1, f0)
        b2 = float(B2[i]) if B2 is not None else hawks_miller_bw(f2, f0)
        b3 = float(B3[i]) if B3 is not None else hawks_miller_bw(f3, f0)
        H1_corr = float(H1[i])
        if not np.isnan(f1) and not np.isnan(b1):
            H1_corr -= iseli_correction(f0, f1, b1, fs)
        if not np.isnan(f2) and not np.isnan(b2):
            H1_corr -= iseli_correction(f0, f2, b2, fs)
        A1_corr = float(A1[i])
        if not np.isnan(f1) and not np.isnan(b1):
            A1_corr -= iseli_correction(f1, f1, b1, fs)
        A2_corr = float(A2[i])
        if not np.isnan(f2) and not np.isnan(b2):
            A2_corr -= iseli_correction(f2, f2, b2, fs)
        A3_corr = float(A3[i])
        if not np.isnan(f1) and not np.isnan(b1):
            A3_corr -= iseli_correction(f3, f1, b1, fs)
        if not np.isnan(f2) and not np.isnan(b2):
            A3_corr -= iseli_correction(f3, f2, b2, fs)
        if not np.isnan(f3) and not np.isnan(b3):
            A3_corr -= iseli_correction(f3, f3, b3, fs)
        H1A1c[i] = H1_corr - A1_corr
        H1A2c[i] = H1_corr - A2_corr
        H1A3c[i] = H1_corr - A3_corr
    return {"H1A1c": H1A1c, "H1A2c": H1A2c, "H1A3c": H1A3c}


def compute_H1H2_H2H4_corrected(
    H1: np.ndarray,
    H2: np.ndarray,
    H4: np.ndarray,
    fs: int,
    F0: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    B1: np.ndarray | None = None,
    B2: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    n = int(F0.shape[0])
    out_H1H2c = np.full(n, np.nan, dtype=float)
    out_H2H4c = np.full(n, np.nan, dtype=float)
    for i in range(n):
        f0 = float(F0[i])
        f1 = float(F1[i]) if i < F1.shape[0] else np.nan
        f2 = float(F2[i]) if i < F2.shape[0] else np.nan
        if np.isnan(f0) or f0 <= 0:
            continue
        b1 = float(B1[i]) if B1 is not None else hawks_miller_bw(f1, f0)
        b2 = float(B2[i]) if B2 is not None else hawks_miller_bw(f2, f0)
        h1c = float(H1[i])
        h2c = float(H2[i])
        h4c = float(H4[i])
        if not np.isnan(f1) and not np.isnan(b1):
            h1c -= iseli_correction(f0, f1, b1, fs)
            h2c -= iseli_correction(2.0 * f0, f1, b1, fs)
            h4c -= iseli_correction(4.0 * f0, f1, b1, fs)
        if not np.isnan(f2) and not np.isnan(b2):
            h1c -= iseli_correction(f0, f2, b2, fs)
            h2c -= iseli_correction(2.0 * f0, f2, b2, fs)
            h4c -= iseli_correction(4.0 * f0, f2, b2, fs)
        out_H1H2c[i] = h1c - h2c
        out_H2H4c[i] = h2c - h4c
    return {"H1H2c": out_H1H2c, "H2H4c": out_H2H4c}

def compute_CPP(
    y: np.ndarray,
    fs: int,
    frameshift_ms: int,
    F0: np.ndarray,
    N_periods: int,
    voiced_mask: np.ndarray | None = None,
) -> np.ndarray:
    """计算 Cepstral Peak Prominence（CPP）。"""
    nframes = int(F0.shape[0])
    CPP = np.full(nframes, np.nan, dtype=float)
    N_ms = int(round(fs / 1000.0))
    for k in range(nframes):
        if voiced_mask is not None and k < voiced_mask.shape[0] and not bool(voiced_mask[k]):
            continue
        f0c = float(F0[k])
        if np.isnan(f0c):
            continue
        seg = _segment_for_frame(y, fs, frameshift_ms, k + 1, N_periods, f0c)
        if seg.size == 0:
            continue
        win = np.hamming(seg.size)
        segw = seg * win
        Y = np.fft.fft(segw)
        y_c = np.fft.ifft(np.log(np.abs(Y) + 1e-12)).real
        y_c_db = 10.0 * np.log10(y_c ** 2 + 1e-12)
        y_c_db = y_c_db[: seg.size // 2]
        if N_ms >= y_c_db.size:
            continue
        N0 = fs / f0c
        v = y_c_db[N_ms:]
        if v.size <= 2:
            continue
        dv = np.diff(v)
        peaks = np.where((dv[:-1] >= 0) & (dv[1:] < 0))[0] + 1
        if peaks.size == 0:
            continue
        winlen = int(round(2.0 * N0))
        sel_idx = []
        for idx in peaks:
            if not sel_idx:
                sel_idx.append(int(idx))
                continue
            if idx - sel_idx[-1] < winlen:
                if v[idx] > v[sel_idx[-1]]:
                    sel_idx[-1] = int(idx)
            else:
                sel_idx.append(int(idx))
        peaks_global = np.array(sel_idx, dtype=int) + N_ms
        if peaks_global.size == 0:
            continue
        near_inx = int(np.argmin(np.abs(peaks_global.astype(float) - N0)))
        peak_pos = int(peaks_global[near_inx])
        peak_val = float(y_c_db[peak_pos])
        xfull = np.arange(N_ms, y_c_db.size, dtype=float)
        yfull = y_c_db[N_ms:]
        p = np.polyfit(xfull, yfull, 1)
        base_val = float(np.polyval(p, peak_pos))
        CPP[k] = peak_val - base_val
    return CPP


def compute_HNR(
    y: np.ndarray,
    fs: int,
    frameshift_ms: int,
    F0: np.ndarray,
    N_periods: int,
    bands_hz: List[int] | None = None,
    voiced_mask: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """计算 HNR（de Krom, 1993）。返回各频带的 HNR。
    参考 MATLAB func_GetHNR。
    """
    if bands_hz is None:
        bands_hz = [500, 1500, 2500, 3500]
    nframes = int(F0.shape[0])
    # 键名与 MATLAB 一致：HNR05/HNR15/HNR25/HNR35
    key_map = {500: "HNR05", 1500: "HNR15", 2500: "HNR25", 3500: "HNR35"}
    HNRs = {key_map.get(b, f"HNR{b}"): np.full(nframes, np.nan, dtype=float) for b in bands_hz}
    for k in range(nframes):
        if voiced_mask is not None and k < voiced_mask.shape[0] and not bool(voiced_mask[k]):
            continue
        f0c = float(F0[k])
        if np.isnan(f0c) or f0c <= 0:
            continue
        seg = _segment_for_frame(y, fs, frameshift_ms, k + 1, N_periods, f0c)
        if seg.size == 0:
            continue
        NBins = seg.size
        N0 = int(round(fs / f0c))
        N0_delta = int(round(N0 * 0.1))
        segw = seg * np.hamming(NBins)
        Y = np.fft.fft(segw, n=NBins)
        aY = np.log10(np.abs(Y) + 1e-12)
        ay = np.fft.ifft(aY).real
        # 寻找 rahmonic 并 lifter 移除
        peakinx = []
        max_k = int(np.floor(seg.size / 2.0 / max(N0, 1)))
        for kk in range(1, max_k + 1):
            ct = kk * N0
            l = max(ct - N0_delta, 0)
            r = min(ct + N0_delta, ay.size - 1)
            inx = l + int(np.argmax(np.abs(ay[l : r + 1])))
            peakinx.append(inx)
            # lifter: 将该峰周围置零
            ay[l:r+1] = 0.0
        # 镜像修复
        midL = int(round(seg.size / 2.0)) + 1
        if midL < ay.size:
            fill_len = ay.size - midL
            if fill_len > 0:
                tail = ay[midL - 1 : midL - 1 - fill_len : -1]
                ay[midL:] = tail
        Nap = np.fft.fft(ay).real
        N = Nap.copy()
        Ha = aY - Nap
        Hdelta = f0c / fs * seg.size
        f = Hdelta
        while f < (seg.size / 2.0):
            fstart = int(np.ceil(f - Hdelta))
            fend = int(min(np.round(f), N.size - 1))
            Bdf = float(np.abs(np.minimum.reduce(Ha[max(fstart,0):fend+1])))
            N[max(fstart,0):fend+1] -= Bdf
            f += Hdelta
        H = aY - N
        for b in bands_hz:
            Ef = int(round(b / fs * seg.size))
            if Ef <= 1:
                continue
            h_val = 20.0 * float(np.mean(H[1:Ef]))
            n_val = 20.0 * float(np.mean(N[1:Ef]))
            key = key_map.get(b, f"HNR{b}")
            HNRs[key][k] = h_val - n_val
    return HNRs


def compute_SHR(
    y: np.ndarray,
    fs: int,
    frameshift_ms: int,
    F0: np.ndarray,
    minf0: float,
    maxf0: float,
    progress_cb: Any | None = None,
    voiced_mask: np.ndarray | None = None,
    shr_threshold: float = 0.0,
) -> np.ndarray:
    nf = int(F0.shape[0])
    y0 = y.astype(float)
    if y0.size == 0:
        return np.full(nf, np.nan, dtype=float)
    y0 = y0 - float(np.mean(y0))
    mabs = float(np.max(np.abs(y0)))
    if mabs > 0.0:
        y0 = y0 / mabs
    frame_len_ms = 40.0
    ceiling = 1250.0
    segmentlen = int(round(frame_len_ms * fs / 1000.0))
    timestep = float(frameshift_ms)
    interpolation_depth = 0.5
    fftlen = 1
    while fftlen < int(segmentlen * (1.0 + interpolation_depth)):
        fftlen *= 2
    frequency = fs * np.arange(1, fftlen // 2 + 1) / float(fftlen)
    limit_idx = int(np.searchsorted(frequency, ceiling, side="left"))
    frequency = frequency[:limit_idx]
    if frequency.size < 2:
        return np.full(nf, np.nan, dtype=float)
    logf = np.log2(frequency)
    min_bin = float(logf[-1] - logf[-2])
    N = int(np.floor(ceiling / float(minf0)))
    N -= N % 2
    N = max(N * 4, 2)
    shift = np.log2(N)
    shift_units = int(round(shift / max(min_bin, 1e-12)))
    interp_logf = np.arange(logf[0], logf[-1] + 1e-12, min_bin)
    interp_len = interp_logf.size
    totallen = shift_units + interp_len
    startpos = shift_units + 1 - np.round(np.log2(np.arange(2, N + 1)) / max(min_bin, 1e-12)).astype(int)
    startpos[startpos < 1] = 1
    endpos = startpos + interp_len - 1
    endpos[endpos > totallen] = totallen
    upperbound = int(np.searchsorted(interp_logf, np.log2(maxf0 / 2.0), side="left"))
    lowerbound = int(np.searchsorted(interp_logf, np.log2(minf0 / 2.0), side="left"))
    out = np.full(nf, np.nan, dtype=float)
    win = np.hamming(segmentlen)
    def _twomax(x: np.ndarray, lb: int, ub: int, unitlen: float):
        lb = max(lb, 0)
        ub = min(ub, x.shape[0] - 1)
        if lb > ub:
            return np.array([]), np.array([])
        xs = x[lb:ub + 1]
        mag1 = float(np.max(xs))
        idx1 = int(np.argmax(xs)) + lb
        if mag1 <= 0:
            return np.array([mag1]), np.array([idx1])
        harmonics = 2.0
        LIMIT = 0.0625
        start = idx1 + int(round(np.log2(harmonics - LIMIT) / unitlen))
        end = idx1 + int(round(np.log2(harmonics + LIMIT) / unitlen))
        end = min(end, x.shape[0] - 1, ub)
        if start <= end:
            xs2 = x[start:end + 1]
            if xs2.size > 0:
                mag2 = float(np.max(xs2))
                if mag2 > 0:
                    idx2 = int(np.argmax(xs2)) + start
                    return np.array([mag1, mag2]), np.array([idx1, idx2])
        return np.array([mag1]), np.array([idx1])
    for i in range(nf):
        f0c = float(F0[i])
        if np.isnan(f0c) or f0c <= 0:
            continue
        if voiced_mask is not None and i < voiced_mask.shape[0] and not bool(voiced_mask[i]):
            continue
        center = int(round(((i) * timestep + frame_len_ms / 2.0) * fs / 1000.0))
        start = center - segmentlen // 2
        if start < 0:
            start = 0
        end = start + segmentlen
        if end > y0.size:
            end = y0.size
            start = max(0, end - segmentlen)
        seg = y0[start:end]
        if seg.size != segmentlen:
            continue
        segw = seg * win
        Spectra = np.fft.fft(segw, n=fftlen)
        amplitude = np.abs(Spectra[:fftlen // 2 + 1])
        amplitude = amplitude[1:limit_idx + 1]
        interp_amplitude = np.interp(interp_logf, logf, amplitude)
        interp_amplitude = interp_amplitude - float(np.min(interp_amplitude))
        len_spectrum = interp_amplitude.shape[0]
        shshift = np.zeros((N, totallen), dtype=float)
        shshift[0, totallen - len_spectrum:totallen] = interp_amplitude
        for ii in range(2, N + 1):
            row = ii - 1
            s = int(startpos[row - 1])
            e = int(endpos[row - 1])
            if e >= s:
                l = e - s + 1
                shshift[row, s - 1:s - 1 + l] = interp_amplitude[:l]
        shshift = shshift[:, shift_units:totallen]
        shodd = np.sum(shshift[0:N:2, :], axis=0)
        sheven = np.sum(shshift[1:N:2, :], axis=0)
        difference = sheven - shodd
        mags, idxs = _twomax(difference, lowerbound, upperbound, min_bin)
        if mags.size == 0:
            val = np.nan
        elif mags.size == 1:
            val = 0.0
        else:
            val = float((mags[0] - mags[1]) / (mags[0] + mags[1] + 1e-12))
        out[i] = val
        if progress_cb is not None and (i % max(1, nf // 10) == 0):
            try:
                progress_cb(i / float(nf))
            except Exception:
                pass
    return out

def compute_spectral_slope(
    y: np.ndarray,
    fs: int,
    frameshift_ms: int,
    F0: np.ndarray,
    min_pitch: float = 40.0,
    max_freq: float = 5000.0,
    N_periods: int = 5,
    voiced_mask: np.ndarray | None = None,
) -> np.ndarray:
    """频谱倾斜（dB vs log10(freq) 线性拟合斜率）。"""
    nf = int(F0.shape[0])
    out = np.full(nf, np.nan, dtype=float)
    for i in range(nf):
        f0c = float(F0[i])
        if np.isnan(f0c) or f0c <= 0:
            continue
        if voiced_mask is not None and i < voiced_mask.shape[0] and not bool(voiced_mask[i]):
            continue
        seg = _segment_for_frame(y, fs, frameshift_ms, i + 1, N_periods, f0c)
        if seg.size == 0:
            continue
        win = np.hamming(seg.size)
        Y = np.fft.fft(seg * win)
        mags = 20.0 * np.log10(np.abs(Y[: seg.size // 2]) + 1e-12)
        freqs = np.linspace(0, fs / 2.0, mags.size)
        msk = (freqs > min_pitch) & (freqs < max_freq) & (mags > (np.max(mags) - 50.0))
        if np.count_nonzero(msk) > 10:
            x = np.log10(freqs[msk])
            yv = mags[msk]
            p = np.polyfit(x, yv, 1)
            out[i] = float(p[0])
    return out

def compute_jitter_shimmer(
    y: np.ndarray,
    fs: int,
    frameshift_ms: int,
    window_ms: int,
    voiced_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """基于 Praat PointProcess 的局部 Jitter/Shimmer（按帧窗口）。"""
    try:
        snd = parselmouth.Sound(y.astype(float), sampling_frequency=fs)
        pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)...", 40, 500)
    except Exception:
        return (np.full(int(round(y.size / fs * 1000.0 / frameshift_ms)), np.nan, dtype=float),
                np.full(int(round(y.size / fs * 1000.0 / frameshift_ms)), np.nan, dtype=float))
    nf = int(round(y.size / float(fs) * 1000.0 / float(frameshift_ms)))
    jit = np.full(nf, np.nan, dtype=float)
    shim = np.full(nf, np.nan, dtype=float)
    for i in range(nf):
        if voiced_mask is not None and i < voiced_mask.shape[0] and not bool(voiced_mask[i]):
            continue
        t = (i + 1) * frameshift_ms / 1000.0
        w = window_ms / 1000.0
        t_s = max(0.0, t - w / 2.0)
        t_e = min(float(y.size) / fs, t + w / 2.0)
        try:
            j = parselmouth.praat.call(pp, "Get jitter (local)", t_s, t_e, 0.0001, 0.02, 1.3)
            s = parselmouth.praat.call([snd, pp], "Get shimmer (local)", t_s, t_e, 0.0001, 0.02, 1.3, 1.6)
            jit[i] = float(j) if j is not None else np.nan
            shim[i] = float(s) if s is not None else np.nan
        except Exception:
            pass
    return jit, shim


def compute_creaky_f0(
    y: np.ndarray,
    fs: int,
    frameshift_ms: int,
) -> np.ndarray:
    nframes = int(round(y.size / float(fs) * 1000.0 / float(frameshift_ms)))
    return np.full(nframes, np.nan, dtype=float)
