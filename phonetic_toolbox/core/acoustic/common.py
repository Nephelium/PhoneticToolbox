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
    # Note: k is 0-based index here, so the center sample is (k + 0.5)? 
    # Original code used (k+1) * sampleshift as center if k was 0-based loop but represented 1-based frame index?
    # In original `workers.py`: ks = int(round((k + 1) * sampleshift)) where k is 0..nf-1
    # In original `praat_service.py`: `_segment_for_frame` takes `k` and does `ks = int(round(k * sampleshift))` 
    # BUT it was called with `k+1` in loops. 
    # Let's standardize: The center of frame k (0-indexed) is usually `k * hop_size + window_size/2` or just `k * hop_size`.
    # VoiceSauce usually aligns centers.
    # Let's stick to the logic: center = (k + 1) * sampleshift to match previous behavior if we pass k as 0-based index?
    # Wait, `praat_service.py` `_segment_for_frame` implementation:
    # def _segment_for_frame(..., k, ...): ks = int(round(k * sampleshift))
    # Call sites: `_segment_for_frame(..., k + 1, ...)` where k is 0..n-1.
    # So effectively center is `(k+1) * sampleshift`.
    
    ks = int(round((k + 1) * sampleshift))
    
    N0 = fs / float(f0_curr)
    half_len = (N_periods / 2.0) * N0
    ystart = int(round(ks - half_len))
    yend = int(round(ks + half_len))
    
    if ystart < 0 or yend > len(y): # Changed >= to > because yend is exclusive
        return np.array([], dtype=float)
        
    seg = y[ystart:yend].astype(float)
    return seg
