import numpy as np
from typing import Optional
from .common import segment_for_frame

def compute_cpp(
    y: np.ndarray,
    fs: int,
    frameshift_ms: float,
    F0: np.ndarray,
    N_periods: int,
    voiced_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    计算倒谱峰值显著度 (Cepstral Peak Prominence, CPP)。
    """
    nframes = int(F0.shape[0])
    CPP = np.full(nframes, np.nan, dtype=float)
    N_ms = int(round(fs / 1000.0)) # 1 ms in samples
    for k in range(nframes):
        if voiced_mask is not None and k < voiced_mask.shape[0] and not bool(voiced_mask[k]):
            continue
        f0c = float(F0[k])
        if np.isnan(f0c) or f0c <= 0:
            continue
        seg = segment_for_frame(y, fs, frameshift_ms, k, N_periods, f0c)
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
