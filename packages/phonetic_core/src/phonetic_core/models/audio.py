"""Decoded samples at original scale; file decoding is an adapter responsibility.

M01-B source_ids: SRC-PRAAT, SRC-REAPER. Preserve separate historical
SciPy/REAPER/Praat conversion semantics instead of normalizing them together.
"""
from dataclasses import dataclass
import numpy as np
import parselmouth


@dataclass(frozen=True)
class AudioInput:
    samples: np.ndarray  # frames, or frames x channels; decoded WAV dtype/scale
    sample_rate_hz: int

    def __post_init__(self):
        data = np.asarray(self.samples)
        if data.ndim not in (1, 2) or (data.ndim == 2 and data.shape[1] == 0):
            raise ValueError('audio must be frames or frames x channels')
        if data.dtype not in (np.dtype('uint8'), np.dtype('int16'), np.dtype('int32'),
                              np.dtype('float32'), np.dtype('float64')):
            raise ValueError('unsupported decoded sample dtype')
        if not np.isfinite(data).all():
            raise ValueError('audio samples must be finite')
        if isinstance(self.sample_rate_hz, bool) or not isinstance(self.sample_rate_hz, int) or self.sample_rate_hz <= 0:
            raise ValueError('sample_rate_hz must be a positive integer')
        data = data.copy()
        data.flags.writeable = False
        object.__setattr__(self, 'samples', data)

    def normalized_channels(self):
        data = self.samples
        if data.dtype == np.int16:
            return data.astype(np.float64) / 32768.0
        if data.dtype == np.int32:
            return data.astype(np.float64) / 2147483648.0
        if data.dtype == np.uint8:
            return (data.astype(np.float64) - 128) / 128.0
        return data.astype(np.float64)

    def scientific_mono(self):
        y = self.normalized_channels()
        return np.mean(y, axis=1) if y.ndim > 1 else y

    def praat_sound(self):
        # Praat's file decoder retains channels; do not supply SciPy's mono mean.
        y = self.normalized_channels()
        return parselmouth.Sound(y.T, sampling_frequency=float(self.sample_rate_hz))
