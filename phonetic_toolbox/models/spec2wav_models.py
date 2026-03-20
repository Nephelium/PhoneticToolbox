from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class Spec2WavConfig:
    """Configuration for Spectrogram to Waveform conversion."""
    image_path: str = ""
    # Or provide image directly if captured from screen
    image_data: Optional[np.ndarray] = None 
    
    time_start: float = 0.0
    time_end: float = 1.0
    freq_start: float = 0.0
    freq_end: float = 11025.0 # Nyquist frequency for 22050 Hz
    min_db: float = -30.0
    max_db: float = 0.0
    n_iter: int = 32
    target_sr: int = 44100 # Default target sample rate
    
    # Derived parameters
    sr: int = 22050 # Will be calculated based on freq_end * 2 usually, but can be overridden

@dataclass
class Spec2WavResult:
    """Result of Spectrogram to Waveform conversion."""
    audio: np.ndarray
    sr: int
    spectrogram: np.ndarray # The linear amplitude spectrogram used
    log_spectrogram: np.ndarray # The log spectrogram (dB)
    duration: float
    reconstructed_spectrogram_db: Optional[np.ndarray] = None # The reconstructed spectrogram in dB
