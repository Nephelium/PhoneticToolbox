import numpy as np
from typing import Optional, Any

def compute_shr(
    y: np.ndarray,
    fs: int,
    frameshift_ms: float,
    F0: np.ndarray,
    minf0: float,
    maxf0: float,
    progress_cb: Any = None,
    voiced_mask: Optional[np.ndarray] = None,
    shr_threshold: float = 0.0,
) -> np.ndarray:
    """
    计算次谐波-谐波比 (Subharmonic-to-Harmonic Ratio, SHR)。
    """
    nf = int(F0.shape[0])
    out = np.full(nf, np.nan, dtype=float)
    y0 = y.astype(float)
    if y0.size == 0:
        return out
    y0 = y0 - float(np.mean(y0))
    mabs = float(np.max(np.abs(y0)))
    if mabs > 0.0:
        y0 = y0 / mabs
    frame_len_ms = 40.0
    ceiling = 1250.0
    segmentlen = int(round(frame_len_ms * fs / 1000.0))
    timestep = float(frameshift_ms)
    
    # FFT length
    fftlen = 1
    while fftlen < int(segmentlen * 1.5):
        fftlen *= 2
    
    frequency = fs * np.arange(1, fftlen // 2 + 1) / float(fftlen)
    limit_idx = int(np.searchsorted(frequency, ceiling, side="left"))
    frequency = frequency[:limit_idx]
    if frequency.size < 2:
        return out
        
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
    win = np.hamming(segmentlen)
    
    for i in range(nf):
        if voiced_mask is not None and i < voiced_mask.shape[0] and not bool(voiced_mask[i]):
            continue
        
        # Center of frame
        center_ms = i * timestep + frame_len_ms / 2.0
        center = int(round(center_ms * fs / 1000.0))
        
        start = center - segmentlen // 2
        start = max(0, start)
        end = start + segmentlen
        if end > y0.size:
            end = y0.size
            start = max(0, end - segmentlen)
        seg = y0[start:end]
        if seg.size != segmentlen:
            continue
            
        # Use F0-guided SHR calculation: Subharmonic Energy / Harmonic Energy
        # F0[i] is the pitch. We look for energy at 0.5*F0 and F0.
        f0_val = float(F0[i])
        if np.isnan(f0_val) or f0_val < minf0 or f0_val > maxf0:
            out[i] = np.nan
            continue

        segw = seg * win
        Spectra = np.fft.rfft(segw, n=fftlen)
        amplitude = np.abs(Spectra)
        freqs = np.fft.rfftfreq(fftlen, d=1.0/fs)
        
        # Helper to find peak amplitude in a frequency range
        def get_peak_amp(target_freq, search_range_ratio=0.1):
            f_min = target_freq * (1 - search_range_ratio)
            f_max = target_freq * (1 + search_range_ratio)
            # Find indices
            idx_start = int(np.searchsorted(freqs, f_min))
            idx_end = int(np.searchsorted(freqs, f_max))
            if idx_start >= idx_end:
                return 0.0
            
            # Find max in range
            return float(np.max(amplitude[idx_start:idx_end]))

        # Harmonic amplitude (at F0)
        harm_amp = get_peak_amp(f0_val)
        
        # Subharmonic amplitude (at 0.5 * F0)
        sub_amp = get_peak_amp(0.5 * f0_val)
        
        if harm_amp <= 0:
            out[i] = 0.0 # No harmonic energy -> 0 SHR? Or NaN? 0 seems safe.
        else:
            # SHR = Sub / Harm
            # Usually 0.x
            ratio = sub_amp / harm_amp
            out[i] = min(1.0, ratio)
            
        if progress_cb:
             progress_cb(i / nf)
             
    return out
