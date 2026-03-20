import numpy as np
from typing import Optional
from .common import segment_for_frame

def compute_spectral_slope(
    y: np.ndarray,
    fs: int,
    frameshift_ms: float,
    F0: np.ndarray,
    min_pitch: float = 40.0,
    max_freq: float = 5000.0,
    N_periods: int = 5,
    voiced_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """计算频谱斜率 (dB vs log10(freq) 线性回归斜率)。"""
    nf = int(F0.shape[0])
    out = np.full(nf, np.nan, dtype=float)
    for i in range(nf):
        if voiced_mask is not None and i < voiced_mask.shape[0] and not bool(voiced_mask[i]):
            continue
        f0c = float(F0[i])
        seg = segment_for_frame(y, fs, frameshift_ms, i, N_periods, f0c)
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
