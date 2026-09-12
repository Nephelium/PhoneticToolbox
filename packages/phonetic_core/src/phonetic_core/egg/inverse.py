"""Simplified closed-phase inverse filter, source PENDING-EGG."""
from ._inverse_legacy import apply_simplified_cp_inverse_filtering
from .errors import EggError, check_cancel, sample_rate, vector
import numbers
import numpy as np


def inverse_filter(audio, fs, gci_times, *, lp_order=None, cancel_event=None):
    check_cancel(cancel_event)
    sample_rate(fs)
    audio = vector(audio)
    events = vector(gci_times, allow_empty=True)
    if np.any(events < 0) or np.any(events >= len(audio)/fs) or np.any(np.diff(events) < 0):
        raise EggError('invalid_gci_grid')
    if lp_order is not None and (isinstance(lp_order, bool) or not isinstance(lp_order, numbers.Integral) or lp_order < 1):
        raise EggError('invalid_inverse_order')
    result = apply_simplified_cp_inverse_filtering(audio, fs, events, lp_order=lp_order, cancel_event=cancel_event)
    check_cancel(cancel_event)
    if result is None or not np.isfinite(result).all():
        raise EggError('inverse_unavailable')
    return result
