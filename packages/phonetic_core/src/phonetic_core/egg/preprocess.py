"""Array adaptation of EGGAnalysisService.load_file (PENDING-EGG)."""
import numpy as np
from scipy import signal

from .config import EGGConfig
from .filters import apply_highpass_filter, apply_lowpass_filter
from .model import EGGAnalysisResult


def prepare(samples: np.ndarray, fs: int, config: EGGConfig, *, flip_channels=False):
    """Own the samples and preserve v2 float32 normalization/filter order."""
    data = np.array(samples, copy=True)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError('EGG requires exactly two channels')
    if len(data) == 0 or data.dtype.kind not in 'iuf' or not np.isfinite(data).all():
        raise ValueError('EGG requires nonempty finite numerical samples')
    if isinstance(fs, bool) or not isinstance(fs, (int, np.integer)) or fs <= 0:
        raise ValueError('Sample rate must be a positive integer')
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    for index in (0, 1):
        peak = float(np.max(np.abs(data[:, index])))
        if peak > 0:
            data[:, index] = (data[:, index] / peak) * .7
    # Preserve v2's interleaved channel layout in our owned data.
    egg = data[:, 1 if flip_channels else 0]
    audio = data[:, 0 if flip_channels else 1]
    detrended = signal.detrend(egg)
    high = apply_highpass_filter(detrended, config.highpass_cutoff, fs)
    processed = apply_lowpass_filter(high, config.lowpass_cutoff, fs)
    times = np.arange(len(processed), dtype=float) / float(fs)
    return EGGAnalysisResult(time_vector=times, egg_signal_raw=egg,
                             egg_signal_processed=processed, audio_signal=audio,
                             fs=int(fs), file_duration=float(times[-1]))
