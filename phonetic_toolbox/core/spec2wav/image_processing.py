import numpy as np
import cv2
from typing import Tuple, Optional

MAX_IMAGE_PIXELS = 25_000_000

def load_spectrogram_image(image_path: str = None, image_data: np.ndarray = None, time_end: float = 1.0, freq_end: float = 5000.0, time_start: float = 0, freq_start: float = 0, min_dB: float = -30.0, max_dB: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Extract spectral data from a spectrogram image and scale it according to time and frequency ranges.
    
    Args:
        image_path (str): Path to the spectrogram image.
        image_data (np.ndarray): Image data (grayscale or color). If provided, image_path is ignored.
        time_start (float): Start time of the spectrogram (seconds), default 0.
        time_end (float): End time of the spectrogram (seconds).
        freq_start (float): Start frequency of the spectrogram (Hz), default 0.
        freq_end (float): End frequency of the spectrogram (Hz).
        min_dB (float): Minimum dB value (corresponding to grayscale 255).
        max_dB (float): Maximum dB value (corresponding to grayscale 0).
        
    Returns:
        Tuple containing:
            - img (np.ndarray): Original image data (grayscale).
            - linear_spectrogram (np.ndarray): Linear amplitude spectrogram.
            - log_spectrogram (np.ndarray): Logarithmic spectrogram (dB).
            - hop_length (int): Calculated hop length.
            - n_fft (int): Calculated FFT size.
            
    Raises:
        ValueError: If image cannot be loaded.
    """
    if time_end <= time_start:
        raise ValueError("time_end must be greater than time_start")
    if freq_end <= freq_start:
        raise ValueError("freq_end must be greater than freq_start")
    if min_dB >= max_dB:
        raise ValueError("min_dB must be less than max_dB")

    img = None
    if image_data is not None:
        if image_data.size == 0:
            raise ValueError("Image data is empty")
        if len(image_data.shape) == 3:
            img = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        elif len(image_data.shape) == 2:
            img = image_data
        else:
            raise ValueError("Image data must be a 2D grayscale or 3D color array")
    elif image_path:
        # Read image as grayscale (assuming brightness represents frequency intensity)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Unable to load image")

    # Get image dimensions
    img_height, img_width = img.shape
    if img_height < 2 or img_width < 1:
        raise ValueError("Image is too small to derive a spectrogram")
    if img_height * img_width > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image is too large ({img_height * img_width} pixels); "
            f"maximum supported size is {MAX_IMAGE_PIXELS} pixels"
        )

    # Flip image vertically to match spectrogram orientation (low freq at bottom)
    img_flipped = np.flipud(img)

    # Map grayscale values to dB: Darker (0) is stronger (max_dB), Lighter (255) is weaker (min_dB)
    log_spectrogram = max_dB - (img_flipped / 255.0) * (max_dB - min_dB) 

    # Convert dB to linear amplitude
    linear_spectrogram = 10 ** (log_spectrogram / 10.0) * 10 

    # Calculate hop_length to match audio duration with image width
    duration = time_end - time_start
    if duration <= 0:
        duration = 1.0 # Fallback
        
    # Set sampling rate to 2 * max frequency (Nyquist theorem)
    sr = int(2 * freq_end)
    
    # Calculate hop_length
    hop_length = max(1, int(duration / img_width * sr))

    # Calculate n_fft to match spectrogram height
    # Spectrogram height is n_fft // 2 + 1
    # So n_fft = 2 * (height - 1)
    n_fft = 2 * (img_height - 1)

    return img, linear_spectrogram, log_spectrogram, hop_length, n_fft
