"""M03 local migration of v2 EGG behavior. Source: PENDING-EGG; provenance unresolved.
See NOTICE.txt. Pure numerical compatibility layer, no file/device/GUI access.
"""
import numpy as np
from scipy import signal
import warnings
from .errors import EggError

def apply_highpass_filter(data: np.ndarray, cutoff_freq: float, fs: int, order: int = 4) -> np.ndarray:
    """
    Apply a high-pass Butterworth filter to the data.

    Args:
        data: Input signal array.
        cutoff_freq: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Order of the filter.

    Returns:
        Filtered signal array.
    """
    nyq = 0.5 * fs
    cutoff = cutoff_freq / nyq
    cutoff = max(cutoff, 1e-6)
    cutoff = min(cutoff, 1 - 1e-6)

    if cutoff >= 1.0:
        warnings.warn(f"High-pass cutoff frequency ({cutoff_freq} Hz) is too high relative to Nyquist ({nyq} Hz). Skipping filtering.")
        return data

    try:
        b, a = signal.butter(order, cutoff, btype='high')
        y = signal.filtfilt(b, a, data)
        return y
    except ValueError as e:
        raise EggError('filter_failed') from e

def apply_lowpass_filter(data: np.ndarray, cutoff_freq: float, fs: int, order: int = 4) -> np.ndarray:
    """
    Apply a low-pass Butterworth filter to the data.

    Args:
        data: Input signal array.
        cutoff_freq: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Order of the filter.

    Returns:
        Filtered signal array.
    """
    nyq = 0.5 * fs
    cutoff = cutoff_freq / nyq
    cutoff = max(cutoff, 1e-6)
    cutoff = min(cutoff, 1 - 1e-6)

    if cutoff <= 0.0:
        warnings.warn(f"Low-pass cutoff frequency ({cutoff_freq} Hz) is too low relative to Nyquist ({nyq} Hz). Skipping filtering.")
        return data

    try:
        b, a = signal.butter(order, cutoff, btype='low')
        y = signal.filtfilt(b, a, data)
        return y
    except ValueError as e:
        raise EggError('filter_failed') from e
