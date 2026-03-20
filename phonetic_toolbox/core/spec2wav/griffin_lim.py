import numpy as np
from scipy import signal
from numpy.fft import fft, ifft
import traceback

# ============================================================================
# Pure NumPy/SciPy implementation of Griffin-Lim algorithm
# This avoids the numba/llvmlite dependency that causes DLL loading issues
# when packaged with PyInstaller.
# ============================================================================

def _stft(y: np.ndarray, n_fft: int, hop_length: int, win_length: int, window: np.ndarray) -> np.ndarray:
    """
    Short-time Fourier Transform using pure NumPy/SciPy.
    
    Parameters:
        y: Input signal
        n_fft: FFT size
        hop_length: Number of samples between frames
        win_length: Window length
        window: Window function array
        
    Returns:
        Complex STFT matrix
    """
    try:
        # Safety checks
        if n_fft < 2:
            n_fft = 2
        if hop_length < 1:
            hop_length = 1
        if len(y) == 0:
            return np.zeros((n_fft // 2 + 1, 1), dtype=np.complex128)
        
        # Pad signal
        pad_length = n_fft // 2
        y_padded = np.pad(y, (pad_length, pad_length), mode='reflect')
        
        # Calculate number of frames
        n_frames = max(1, 1 + (len(y_padded) - n_fft) // hop_length)
        
        # Create output matrix
        stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
        
        # Pad window to n_fft if needed
        if len(window) < n_fft:
            window_padded = np.zeros(n_fft)
            start = (n_fft - len(window)) // 2
            window_padded[start:start + len(window)] = window
            window = window_padded
        elif len(window) > n_fft:
            window = window[:n_fft]
        
        for i in range(n_frames):
            start = i * hop_length
            end = start + n_fft
            if end > len(y_padded):
                break
            frame = y_padded[start:end] * window
            spectrum = fft(frame)
            stft_matrix[:, i] = spectrum[:n_fft // 2 + 1]
        
        return stft_matrix
    except Exception as e:
        print(f"[_stft] Error: {e}")
        traceback.print_exc()
        raise


def _istft(stft_matrix: np.ndarray, hop_length: int, win_length: int, n_fft: int, window: np.ndarray, length: int = None) -> np.ndarray:
    """
    Inverse Short-time Fourier Transform using pure NumPy/SciPy.
    
    Parameters:
        stft_matrix: Complex STFT matrix
        hop_length: Number of samples between frames
        win_length: Window length
        n_fft: FFT size
        window: Window function array
        length: Expected output length (optional)
        
    Returns:
        Reconstructed signal
    """
    try:
        n_frames = stft_matrix.shape[1]
        
        # Safety check for n_fft
        if n_fft < 2:
            n_fft = 2
        
        # Reconstruct full spectrum (conjugate symmetric)
        full_spectrum = np.zeros((n_fft, n_frames), dtype=np.complex128)
        
        # Safely copy the STFT matrix
        n_bins = min(n_fft // 2 + 1, stft_matrix.shape[0])
        full_spectrum[:n_bins, :] = stft_matrix[:n_bins, :]
        
        # Fill conjugate symmetric part
        if n_fft > 2:
            conj_start = n_fft // 2 + 1
            conj_end = n_fft
            src_start = n_bins - 2
            src_end = 0
            if src_start > src_end:
                full_spectrum[conj_start:conj_end, :] = np.conj(stft_matrix[src_start:src_end:-1, :])
        
        # Pad window to n_fft if needed
        if len(window) < n_fft:
            window_padded = np.zeros(n_fft)
            start = (n_fft - len(window)) // 2
            window_padded[start:start + len(window)] = window
            window = window_padded
        elif len(window) > n_fft:
            window = window[:n_fft]
        
        # Calculate output length
        expected_length = n_fft + hop_length * (n_frames - 1)
        y = np.zeros(expected_length)
        window_sum = np.zeros(expected_length)
        
        for i in range(n_frames):
            start = i * hop_length
            end = start + n_fft
            if end > expected_length:
                break
            frame = np.real(ifft(full_spectrum[:, i]))
            y[start:end] += frame * window
            window_sum[start:end] += window ** 2
        
        # Normalize by window sum (avoid division by zero)
        window_sum = np.maximum(window_sum, 1e-8)
        y = y / window_sum
        
        # Remove padding safely
        pad_length = n_fft // 2
        if pad_length > 0 and len(y) > 2 * pad_length:
            y = y[pad_length:-pad_length]
        
        if length is not None:
            if len(y) > length:
                y = y[:length]
            elif len(y) < length:
                y = np.pad(y, (0, length - len(y)))
        
        return y
    except Exception as e:
        print(f"[_istft] Error: {e}")
        raise


def griffinlim_numpy(S: np.ndarray, n_iter: int = 32, hop_length: int = 512, win_length: int = None, n_fft: int = 2048) -> np.ndarray:
    """
    Griffin-Lim algorithm for phase reconstruction using pure NumPy/SciPy.
    
    This implementation avoids the numba/llvmlite dependency that causes
    DLL loading issues when packaged with PyInstaller.
    """
    if win_length is None:
        win_length = n_fft
    
    # Ensure win_length <= n_fft
    win_length = min(win_length, n_fft)
    
    # Create Hann window
    window = signal.windows.hann(win_length, sym=False)
    
    # Initialize with random phase
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = S * angles
    
    # Estimate output length
    n_frames = S.shape[1]
    length = hop_length * (n_frames - 1) + n_fft - 2 * (n_fft // 2)
    
    # Griffin-Lim iterations
    for _ in range(n_iter):
        # Inverse STFT
        y = _istft(S_complex, hop_length, win_length, n_fft, window, length)
        
        # Forward STFT
        S_rebuilt = _stft(y, n_fft, hop_length, win_length, window)
        
        # Update phase while keeping magnitude
        angles = np.exp(1j * np.angle(S_rebuilt))
        S_complex = S * angles
    
    # Final inverse STFT
    y = _istft(S_complex, hop_length, win_length, n_fft, window, length)
    
    return y

def spectrogram_to_audio(spectrogram: np.ndarray, hop_length: int, window_length: int, n_fft: int, sr: int = 22050, n_iter: int = 32) -> np.ndarray:
    """
    Convert spectrogram to audio using Griffin-Lim algorithm.
    Wrapper around griffinlim_numpy.
    
    Args:
        spectrogram (np.ndarray): Spectrogram matrix.
        hop_length (int): Hop length.
        window_length (int): Window length.
        n_fft (int): FFT size.
        sr (int): Sample rate (default 22050).
        n_iter (int): Number of Griffin-Lim iterations (default 32).
    
    Returns:
        np.ndarray: Reconstructed audio signal.
    """
    return griffinlim_numpy(spectrogram, n_iter=n_iter, hop_length=hop_length, win_length=window_length, n_fft=n_fft)
