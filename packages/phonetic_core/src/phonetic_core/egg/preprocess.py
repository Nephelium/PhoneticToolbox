"""Array adaptation of EGGAnalysisService.load_file (PENDING-EGG)."""
import numpy as np
from scipy import signal

from .config import EGGConfig
from .filters import apply_highpass_filter, apply_lowpass_filter
from .model import EGGAnalysisResult
from .errors import EggError, check_cancel, cutoffs


def prepare(samples: np.ndarray, fs: int, config: EGGConfig, *, flip_channels=False, cancel_event=None):
    """Own the samples and preserve v2 float32 normalization/filter order."""
    check_cancel(cancel_event)
    cutoffs(config, fs)
    if type(flip_channels) is not bool:
        raise EggError('invalid_channel_selection')
    data = np.array(samples, copy=True, order='C')
    if data.ndim != 2 or data.shape[1] != 2:
        raise EggError('requires_two_channels')
    if len(data) == 0 or data.dtype.kind not in 'iuf' or not np.isfinite(data).all():
        raise EggError('invalid_signal')
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        if np.max(np.abs(data)) > np.finfo(np.float32).max:
            raise EggError('signal_float32_overflow')
        data = data.astype(np.float32)
    if not np.isfinite(data).all():
        raise EggError('signal_float32_overflow')
    for index in (0, 1):
        peak = float(np.max(np.abs(data[:, index])))
        if peak > 0:
            data[:, index] = (data[:, index] / peak) * .7
    # Preserve v2's interleaved channel layout in our owned data.
    egg = data[:, 1 if flip_channels else 0]
    audio = data[:, 0 if flip_channels else 1]
    detrended = signal.detrend(egg)
    high = apply_highpass_filter(detrended, config.highpass_cutoff, fs)
    check_cancel(cancel_event)
    processed = apply_lowpass_filter(high, config.lowpass_cutoff, fs)
    check_cancel(cancel_event)
    times = np.arange(len(processed), dtype=float) / float(fs)
    return EGGAnalysisResult(time_vector=times, egg_signal_raw=egg,
                             egg_signal_processed=processed, audio_signal=audio,
                             fs=int(fs), file_duration=float(times[-1]), preprocessing_config=config)
