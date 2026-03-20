import numpy as np
import sys

def resample_audio(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio using linear interpolation.
    
    This implementation uses pure NumPy to avoid scipy.signal.resample
    which can cause crashes in PyInstaller frozen apps due to FFT/BLAS issues.
    
    Parameters:
        y: Input audio signal
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio signal
    """
    if orig_sr == target_sr:
        return y
    
    # Calculate new length
    duration = len(y) / orig_sr
    new_length = int(duration * target_sr)
    
    # Use linear interpolation (pure NumPy, no scipy dependency)
    old_indices = np.arange(len(y))
    new_indices = np.linspace(0, len(y) - 1, new_length)
    y_resampled = np.interp(new_indices, old_indices, y)
    
    return y_resampled


def amplitude_to_db(S: np.ndarray, ref=1.0, amin=1e-10, top_db=80.0) -> np.ndarray:
    """
    Convert amplitude spectrogram to dB-scaled spectrogram.
    
    Parameters:
        S: Input amplitude spectrogram
        ref: Reference value for dB conversion (scalar or callable)
        amin: Minimum amplitude threshold
        top_db: Maximum dB range
        
    Returns:
        dB-scaled spectrogram
    """
    S = np.asarray(S)
    
    if callable(ref):
        ref_value = ref(S)
    else:
        ref_value = np.abs(ref)
    
    log_spec = 10.0 * np.log10(np.maximum(amin, S))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    
    return log_spec
