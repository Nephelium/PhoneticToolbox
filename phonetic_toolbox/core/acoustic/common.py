import numpy as np

def segment_for_frame(y: np.ndarray, fs: int, frameshift_ms: float, k: int, N_periods: float, f0_curr: float) -> np.ndarray:
    """
    Slice a segment of speech for the k-th frame with a length of N_periods pitch periods.
    
    Args:
        y: Audio signal array.
        fs: Sampling frequency (Hz).
        frameshift_ms: Frame shift in milliseconds.
        k: Frame index (0-based).
        N_periods: Number of pitch periods to include in the segment.
        f0_curr: Current F0 value (Hz).

    Returns:
        Segmented audio array. Returns empty array if F0 is invalid or segment is out of bounds.
    """
    if np.isnan(f0_curr) or f0_curr <= 0:
        return np.array([], dtype=float)
    
    sampleshift = int(round(fs / 1000.0 * frameshift_ms))
    # Frame k corresponds to the canonical zero-based grid time
    # ``k * frameshift_ms`` used by the service layer.
    ks = int(round(k * sampleshift))
    
    N0 = fs / float(f0_curr)
    half_len = (N_periods / 2.0) * N0
    ystart = int(round(ks - half_len))
    yend = int(round(ks + half_len))
    
    if ystart < 0 or yend > len(y): # Changed >= to > because yend is exclusive
        return np.array([], dtype=float)
        
    seg = y[ystart:yend].astype(float)
    return seg
