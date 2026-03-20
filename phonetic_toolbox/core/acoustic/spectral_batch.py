import numpy as np
from typing import Dict, Optional, Tuple, List
from scipy.signal import get_window
from .common import segment_for_frame

def parabolic_interpolation(y: np.ndarray, x: np.ndarray, i: int) -> Tuple[float, float]:
    """
    Perform parabolic interpolation to find the peak position and value.
    
    Args:
        y: Amplitude values (e.g., in dB)
        x: Frequency values (Hz)
        i: Index of the peak in y
        
    Returns:
        (peak_val, peak_freq)
    """
    if i <= 0 or i >= len(y) - 1:
        return y[i], x[i]
        
    alpha = y[i-1]
    beta = y[i]
    gamma = y[i+1]
    
    # Peak offset calculation (based on quadratic fit)
    denom = 2 * (alpha - 2 * beta + gamma)
    if denom == 0:
        return beta, x[i]
        
    p = 0.5 * (alpha - gamma) / denom
    
    # Peak value
    peak_val = beta - 0.25 * (alpha - gamma) * p
    
    # Peak frequency
    # Assuming x is uniformly spaced, freq = x[i] + p * (x[i+1] - x[i])
    # But x might not be uniform if we passed arbitrary grid, though usually FFT bins are.
    dx = x[1] - x[0] # Assume uniform spacing
    peak_freq = x[i] + p * dx
    
    return peak_val, peak_freq

def compute_spectral_features_batch(
    y: np.ndarray,
    fs: int,
    frameshift_ms: float,
    F0: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    F3: np.ndarray,
    N_periods: int,
    voiced_mask: Optional[np.ndarray] = None,
    fft_pad_factor: int = 4, # Legacy param, ignored if target_resolution is set
    target_resolution: float = 0.5 # Default resolution for optimal speed/accuracy balance
) -> Dict[str, np.ndarray]:
    """
    Batch compute spectral features: H1, H2, H4, A1, A2, A3, H2K, H5K.
    
    Optimized approach:
    1. For each frame, extract segment ONCE.
    2. Compute FFT (with zero padding) ONCE.
    3. Look up all harmonics in the same spectrum.
    4. Use parabolic interpolation for precise peak estimation.
    """
    nframes = len(F0)
    
    # Initialize output arrays
    out = {
        "H1": np.full(nframes, np.nan),
        "H2": np.full(nframes, np.nan),
        "H4": np.full(nframes, np.nan),
        "A1": np.full(nframes, np.nan),
        "A2": np.full(nframes, np.nan),
        "A3": np.full(nframes, np.nan),
        "H2K": np.full(nframes, np.nan),
        "H5K": np.full(nframes, np.nan),
    }
    
    # Pre-calculate common constants
    # Using a window function reduces spectral leakage
    # Gaussian window is good for harmonic analysis, but Hamming/Hanning is standard.
    # The original code didn't use a window explicitly in segment_for_frame (it used rectangular),
    # but for FFT based peak picking, a window is recommended.
    # However, to match previous behavior closely, we might stick to what segment_for_frame does.
    # segment_for_frame extracts exactly N_periods. Rectangular window on integer number of periods
    # of a periodic signal is actually optimal (no leakage).
    # Since we extract N_periods based on F0, we assume it's close to integer periods.
    # Let's use a Hanning window to be safe against pitch estimation errors.
    
    for k in range(nframes):
        if voiced_mask is not None and k < len(voiced_mask) and not voiced_mask[k]:
            continue
            
        f0c = float(F0[k])
        if np.isnan(f0c) or f0c <= 0:
            continue
            
        # 1. Extract Segment
        seg = segment_for_frame(y, fs, frameshift_ms, k, N_periods, f0c)
        n_samples = seg.size
        if n_samples == 0:
            continue
            
        # Apply window (optional but recommended for FFT)
        # window = get_window('hamming', n_samples)
        # seg_windowed = seg * window
        # For now, use raw segment to match previous logic (which used rectangular)
        seg_windowed = seg 
        
        # 2. Compute FFT
        # Determine n_fft
        if target_resolution:
            # Resolution df = fs / n_fft  => n_fft = fs / df
            n_fft_target = int(fs / target_resolution)
            # Find next power of 2 >= n_fft_target
            n_fft = 1
            while n_fft < n_fft_target or n_fft < n_samples:
                n_fft *= 2
        else:
            # Zero-pad to next power of 2 * pad_factor for high resolution
            n_fft = 1
            while n_fft < n_samples * fft_pad_factor:
                n_fft *= 2
        
        # Ensure minimum FFT size for low frequency resolution
        # We want resolution better than f0 * 0.1?
        # Bin width = fs / n_fft. We want fs/n_fft < f0 * 0.1 usually.
        # But n_fft is already quite large due to pad_factor.
        
        fft_spec = np.fft.rfft(seg_windowed, n=n_fft)
        mag_spec = np.abs(fft_spec)
        mag_db = 20 * np.log10(mag_spec + 1e-12)
        freqs = np.fft.rfftfreq(n_fft, d=1/fs)
        
        # Bin width
        df = fs / n_fft
        
        # Helper to find peak in range
        def find_peak_in_range(target_freq: float, search_range: float) -> float:
            if np.isnan(target_freq) or target_freq <= 0:
                return np.nan
            
            f_min = target_freq - search_range
            f_max = target_freq + search_range
            
            # Convert to bin indices
            idx_min = int(np.floor(f_min / df))
            idx_max = int(np.ceil(f_max / df))
            
            # Clamp
            idx_min = max(0, idx_min)
            idx_max = min(len(mag_db) - 1, idx_max)
            
            if idx_min >= idx_max:
                return mag_db[idx_min]
                
            # Find max in range
            sub_region = mag_db[idx_min : idx_max + 1]
            if len(sub_region) == 0:
                return np.nan
                
            rel_max_idx = np.argmax(sub_region)
            abs_max_idx = idx_min + rel_max_idx
            
            # Parabolic interpolation
            val, _ = parabolic_interpolation(mag_db, freqs, abs_max_idx)
            return val

        # 3. Look up harmonics
        
        # H1, H2, H4
        out["H1"][k] = find_peak_in_range(f0c, f0c * 0.2)
        out["H2"][k] = find_peak_in_range(2.0 * f0c, f0c * 0.3)
        out["H4"][k] = find_peak_in_range(4.0 * f0c, f0c * 0.4)
        
        # H2K (nearest harmonic to 2000 Hz)
        h_idx_2k = max(1, int(round(2000.0 / f0c)))
        out["H2K"][k] = find_peak_in_range(h_idx_2k * f0c, f0c * 0.5) # Wider range for fixed freq search?
        # Actually previous code used search_peak_mag_db_range with range_hz = f0c * 0.5 for fixed freq
        
        # H5K (nearest harmonic to 5000 Hz)
        h_idx_5k = max(1, int(round(5000.0 / f0c)))
        out["H5K"][k] = find_peak_in_range(h_idx_5k * f0c, f0c * 0.5)
        
        # A1, A2, A3 (nearest harmonic to formants)
        # Helper for A-params
        def get_amp_nearest_harmonic(formant_freq: float) -> float:
            if np.isnan(formant_freq) or formant_freq <= 0:
                return np.nan
            h_idx = max(1, int(round(formant_freq / f0c)))
            target_h_freq = h_idx * f0c
            return find_peak_in_range(target_h_freq, f0c * 0.5) # Previous code used range 0.5 * f0c
            
        out["A1"][k] = get_amp_nearest_harmonic(float(F1[k]))
        out["A2"][k] = get_amp_nearest_harmonic(float(F2[k]))
        out["A3"][k] = get_amp_nearest_harmonic(float(F3[k]))
        
    return out
