import numpy as np
from typing import Dict, List, Optional
from .common import segment_for_frame

def compute_hnr(
    y: np.ndarray,
    fs: int,
    frameshift_ms: float,
    F0: np.ndarray,
    N_periods: int,
    bands_hz: Optional[List[int]] = None,
    voiced_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    计算多频带谐波噪声比 (Harmonics-to-Noise Ratio, HNR)。
    """
    if bands_hz is None:
        bands_hz = [500, 1500, 2500, 3500]
    nframes = int(F0.shape[0])
    key_map = {500: "HNR05", 1500: "HNR15", 2500: "HNR25", 3500: "HNR35"}
    HNRs = {key_map.get(b, f"HNR{b}"): np.full(nframes, np.nan, dtype=float) for b in bands_hz}
    
    for k in range(nframes):
        if voiced_mask is not None and k < voiced_mask.shape[0] and not bool(voiced_mask[k]):
            continue
        f0c = float(F0[k])
        if np.isnan(f0c) or f0c <= 0:
            continue
        seg = segment_for_frame(y, fs, frameshift_ms, k, N_periods, f0c)
        if seg.size == 0:
            continue
            
        NBins = seg.size
        N0 = int(round(fs / f0c))
        N0_delta = int(round(N0 * 0.1))
        segw = seg * np.hamming(NBins)
        Y = np.fft.fft(segw, n=NBins)
        aY = np.log10(np.abs(Y) + 1e-12)
        ay = np.fft.ifft(aY).real
        
        max_k = int(np.floor(seg.size / 2.0 / max(N0, 1)))
        for kk in range(1, max_k + 1):
            ct = kk * N0
            l = max(ct - N0_delta, 0)
            r = min(ct + N0_delta, ay.size - 1)
            ay[l:r+1] = 0.0
            
        midL = int(round(seg.size / 2.0)) + 1
        if midL < ay.size:
            fill_len = ay.size - midL
            if fill_len > 0:
                ay[midL:] = ay[midL - 1 : midL - 1 - fill_len : -1]
                
        Nap = np.fft.fft(ay).real
        N = Nap.copy()
        Ha = aY - Nap
        Hdelta = f0c / fs * seg.size
        f = Hdelta
        while f < (seg.size / 2.0):
            fstart = int(np.ceil(f - Hdelta))
            fend = int(min(np.round(f), N.size - 1))
            if fstart <= fend:
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
