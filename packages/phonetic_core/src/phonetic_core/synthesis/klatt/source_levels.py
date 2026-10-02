"""M06/2 digital source calibration, not sound-pressure calibration.

Klatt (1980), Table I supplies the AV control range, not this digital reference.
At 60 dB the reference voiced signal and theoretical aspiration noise each have
RMS 0.01 before the vocal tract. The voiced reference is the existing tdklatt
glottal pole/zero driven at 120 Hz, with zero jitter, shimmer and diplophonia.
This fixed reference never measures or normalizes the user's utterance.
"""
from functools import lru_cache
import numpy as np
from scipy.signal import lfilter

INTERNAL_RATE = 20000
REFERENCE_DB = 60.0
REFERENCE_RMS = 0.01
OUTPUT_GAIN = 64.0  # Fixed digital monitor/export gain, never automatic gain.
REVISION = 'klatt/2'


def source_gain(db):
    values = np.asarray(db, dtype=float)
    return np.where(values > 0, np.power(10., (values - REFERENCE_DB) / 20.), 0.)


@lru_cache(maxsize=16)
def voice_reference_rms(rate, sinusoidal=False):
    """Independent LTI calibration of tdklatt's fixed default glottal filters."""
    def coefficients(frequency, bandwidth):
        c = -np.exp(-2 * np.pi * bandwidth / rate)
        b = 2 * np.exp(-np.pi * bandwidth / rate) * np.cos(2 * np.pi * frequency / rate)
        return 1 - b - c, b, c
    period = round(rate / 120.)
    signal = np.zeros(period * 256)
    signal[::period] = 1.
    a, b, c = coefficients(0., 100.)
    signal = lfilter([a], [1., -b, -c], signal)
    if sinusoidal:
        a, b, c = coefficients(0., 200.)
        signal = lfilter([a], [1., -b, -c], signal)
    else:
        a, b, c = coefficients(1500., 6000.)
        signal = lfilter([1. / a, -b / a, -c / a], [1.], signal)
    return float(np.sqrt(np.mean(signal[-period * 64:] ** 2)))
