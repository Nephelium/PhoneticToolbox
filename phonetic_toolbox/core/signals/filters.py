import numpy as np
from scipy import signal
import pywt
import warnings
from typing import Optional, Union

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
        warnings.warn(f"Error during high-pass filtering: {e}. Cutoff: {cutoff}. Returning unfiltered data.")
        return data

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
        warnings.warn(f"Error during low-pass filtering: {e}. Cutoff: {cutoff}. Returning unfiltered data.")
        return data

def apply_wavelet_denoising(data: np.ndarray, wavelet: str = 'db4', level: int = 4, mode: str = 'soft') -> np.ndarray:
    """
    Apply wavelet denoising to the data.

    Args:
        data: Input signal array.
        wavelet: Wavelet name (e.g., 'db4').
        level: Decomposition level.
        mode: Thresholding mode ('soft' or 'hard').

    Returns:
        Denoised signal array.
    """
    if data is None or len(data) < 2:
        return data
        
    try:
        coeffs = pywt.wavedec(data, wavelet, level=level)
        detail_coeffs = [c for c in coeffs[1:] if c is not None and len(c) > 0]
        
        if not detail_coeffs:
            warnings.warn("No valid detail coefficients found for noise estimation. Skipping thresholding.")
            threshold = 0
        else:
            last_detail_coeffs = detail_coeffs[-1]
            sigma = np.median(np.abs(last_detail_coeffs - np.median(last_detail_coeffs))) / 0.6745
            if sigma < 1e-9: sigma = np.std(last_detail_coeffs)
            if sigma < 1e-9: sigma = 1e-9
            threshold = sigma * np.sqrt(2 * np.log(len(data))) if len(data) > 1 else 0

        coeffs_thresh = [coeffs[0]]
        for i in range(1, len(coeffs)):
             if coeffs[i] is not None and len(coeffs[i]) > 0:
                coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode=mode))
             else:
                coeffs_thresh.append(coeffs[i])

        denoised_data = pywt.waverec(coeffs_thresh, wavelet)
        
        # Match length
        original_len = len(data)
        current_len = len(denoised_data)
        if current_len > original_len:
            denoised_data = denoised_data[:original_len]
        elif current_len < original_len:
            padding = original_len - current_len
            denoised_data = np.pad(denoised_data, (0, padding), 'edge')

        return denoised_data
    except Exception as e:
        warnings.warn(f"Error during wavelet denoising: {e}. Returning original data.")
        return data
