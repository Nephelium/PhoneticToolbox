"""Playback-only gain policy; synthesis, capture and exports remain unchanged."""
import math
import numpy as np

AUDITION_PEAK = 10 ** (-1 / 20)
MAX_AUDITION_GAIN = 32.


def peak_gain(peak, enabled=True):
    return 1. if not enabled or not math.isfinite(peak) or peak <= 1e-5 else min(MAX_AUDITION_GAIN, AUDITION_PEAK / peak)


def sample_peak(samples):
    values = np.asarray(samples)
    return float(np.max(np.abs(values))) if values.size else 0.
