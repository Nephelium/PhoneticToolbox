# spec2wav module
# Spectrogram to audio conversion using Griffin-Lim algorithm
#
# This module provides tools for converting spectrogram images to audio
# using the Griffin-Lim algorithm with pure NumPy/SciPy implementation.
#
# Usage:
#   from spec2wav.spec2wav_gui import MainWindow
#   from spec2wav.spec2wav import load_spectrogram_image, spectrogram_to_audio

__version__ = "1.0.0"

# Lazy imports to avoid loading heavy dependencies at module import time
# This is important for PyInstaller compatibility

def __getattr__(name):
    """Lazy import of submodules to avoid import errors in frozen apps."""
    if name == 'MainWindow':
        from .spec2wav_gui import MainWindow
        return MainWindow
    elif name == 'load_spectrogram_image':
        from .spec2wav import load_spectrogram_image
        return load_spectrogram_image
    elif name == 'spectrogram_to_audio':
        from .spec2wav import spectrogram_to_audio
        return spectrogram_to_audio
    elif name == 'griffinlim_numpy':
        from .spec2wav import griffinlim_numpy
        return griffinlim_numpy
    elif name == 'process_spectrogram_image':
        from .spec2wav_gui import process_spectrogram_image
        return process_spectrogram_image
    elif name == 'resample_audio':
        from .spec2wav_gui import resample_audio
        return resample_audio
    elif name == 'amplitude_to_db':
        from .spec2wav_gui import amplitude_to_db
        return amplitude_to_db
    raise AttributeError(f"module 'spec2wav' has no attribute '{name}'")

__all__ = [
    'MainWindow',
    'load_spectrogram_image',
    'spectrogram_to_audio', 
    'griffinlim_numpy',
    'process_spectrogram_image',
    'resample_audio',
    'amplitude_to_db',
]
