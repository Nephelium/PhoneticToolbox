import numpy as np

def compute_energy(
    y: np.ndarray,
    fs: int,
    frameshift_ms: float,
    F0: np.ndarray,
    energy_window_ms: float = 20.0,
    Nperiods_EC: float = 5.0, # Not used in Energy calculation but usually part of config
) -> np.ndarray:
    """
    Compute frame-based Energy (Intensity similar to Praat).
    Praat calibration: a sine wave with amplitude 1 has an intensity of 96 dB.
    10*log10(0.5) + C = 96  => C ~= 99.01
    
    Args:
        y: Audio signal.
        fs: Sampling rate.
        frameshift_ms: Frame shift in ms.
        F0: F0 track (used to align frames, although energy is usually window-based).
        energy_window_ms: Window size in ms.
    
    Returns:
        Intensity array in dB (positive values).
    """
    sampleshift = int(round(fs / 1000.0 * frameshift_ms))
    # Praat uses a minimum pitch to determine window size if not specified, 
    # but here we use energy_window_ms (default 20ms is standard for energy).
    # Praat uses 3.2 / min_pitch (e.g. 3.2/100 = 0.032s = 32ms).
    
    win = int(round(fs / 1000.0 * energy_window_ms))
    nf = len(F0)
    Ewin = np.full(nf, np.nan, dtype=float)
    
    # Pre-calculate constant for Praat-like scaling
    # 10 * log10(1/P0) where P0 = 4e-10 is the reference pressure squared?
    # Praat: 10 * log10(mean_power) + 96 + 10*log10(2) ?
    # Let's use the simple calibration: 10*log10(mean_sq) + 96 + 3.0103
    calibration_db = 96.0 + 10 * np.log10(2.0) # ~ 99.01
    
    for k in range(nf):
        s = k * sampleshift
        # Center the window? Praat usually centers.
        # But our system might use left-aligned. Let's stick to existing loop structure but check centering.
        # If F0 analysis is centered, energy should be too.
        # Current loop: s is start.
        # Let's try to center it: s = k * sampleshift - win // 2
        
        # Using centered window for better alignment with F0
        mid_point = int(k * sampleshift)
        start_idx = max(0, mid_point - win // 2)
        end_idx = min(len(y), mid_point + win // 2)
        
        seg = y[start_idx:end_idx]
        
        if seg.size > 0:
            mean_sq = np.mean(seg * seg)
            if mean_sq > 0:
                val_db = 10.0 * np.log10(mean_sq) + calibration_db
                Ewin[k] = val_db
            else:
                Ewin[k] = 0.0 # Silence
        else:
            Ewin[k] = np.nan
            
    return Ewin

def compute_rms(
    y: np.ndarray,
    fs: int,
    frameshift_ms: float,
    F0: np.ndarray,
    window_ms: float = 20.0,
) -> np.ndarray:
    """Compute Root Mean Square (RMS) amplitude."""
    frame_shift = int(round(fs / 1000.0 * frameshift_ms))
    win_samples = int(round(fs / 1000.0 * window_ms))
    if frame_shift <= 0 or win_samples <= 0:
        raise ValueError("frameshift_ms 和 window_ms 必须产生至少一个采样点")

    rms = np.full(len(F0), np.nan, dtype=float)
    for index in range(len(F0)):
        center = index * frame_shift
        start = max(0, center - win_samples // 2)
        end = min(len(y), center + win_samples // 2)
        segment = np.asarray(y[start:end], dtype=float)
        if segment.size:
            rms[index] = float(np.sqrt(np.mean(segment * segment)))
    return rms
