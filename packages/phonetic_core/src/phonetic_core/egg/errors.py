"""Explicit M03 boundary failures. No input paths or sample values in messages."""
import math
import numbers

import numpy as np


class EggError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class EggCancelled(EggError):
    def __init__(self):
        super().__init__('egg_cancelled')


def check_cancel(event):
    if event is not None and event.is_set():
        raise EggCancelled()


def sample_rate(fs):
    if isinstance(fs, bool) or not isinstance(fs, numbers.Integral) or fs <= 0:
        raise EggError('invalid_sample_rate')


def finite_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def vector(value, *, allow_empty=False):
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.dtype.kind not in 'iuf' or not np.isfinite(arr).all() or (not allow_empty and arr.size == 0):
        raise EggError('invalid_signal')
    return arr


def roi(start, end):
    if not finite_number(start) or not finite_number(end) or end < start:
        raise EggError('invalid_roi')


def cutoffs(config, fs):
    sample_rate(fs)
    if not 0 < config.highpass_cutoff < config.lowpass_cutoff < fs/2:
        raise EggError('filter_cutoff')
