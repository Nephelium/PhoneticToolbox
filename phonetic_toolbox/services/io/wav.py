from pathlib import Path
from typing import Tuple, Union
import numpy as np
import scipy.io.wavfile as wavfile

def read_wav(path: Union[str, Path]) -> Tuple[int, np.ndarray]:
    """
    Read a WAV file (raw).
    
    Args:
        path: Path to the WAV file.
        
    Returns:
        Tuple[int, np.ndarray]: Sampling rate and data (int or float depending on file).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"WAV file not found: {p}")
        
    try:
        fs, data = wavfile.read(str(p))
        return fs, data
    except Exception as e:
        raise ValueError(f"Error reading WAV {p}: {e}")

def read_wav_float_mono(path: Union[str, Path]) -> Tuple[int, np.ndarray]:
    """
    Read a WAV file, convert to float (-1.0 to 1.0) and mono.
    
    Args:
        path: Path to the WAV file.
        
    Returns:
        Tuple[int, np.ndarray]: Sampling rate and mono float data.
    """
    fs, data = read_wav(path)
    
    # Convert to float
    if data.dtype == np.int16:
        y = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        y = (data.astype(np.float64) - 128) / 128.0
    else:
        y = data.astype(np.float64)
    
    # Convert to mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)
        
    return fs, y

def write_wav(path: Union[str, Path], fs: int, data: np.ndarray):
    """
    Write data to a WAV file.
    
    Args:
        path: Path to write to.
        fs: Sampling rate.
        data: Audio data.
    """
    p = Path(path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        
    try:
        wavfile.write(str(p), fs, data)
    except Exception as e:
        raise ValueError(f"Error writing WAV {p}: {e}")
