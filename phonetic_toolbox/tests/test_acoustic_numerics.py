from __future__ import annotations

import numpy as np

from phonetic_toolbox.core.acoustic.energy import compute_rms
from phonetic_toolbox.core.acoustic.spectral_batch import parabolic_interpolation
from phonetic_toolbox.services.acoustic_service import smooth_preserving_gaps


def test_parabolic_interpolation_recovers_exact_peak():
    frequencies = np.arange(4, dtype=float)
    magnitudes = -((frequencies - 1.25) ** 2)

    peak_value, peak_frequency = parabolic_interpolation(
        magnitudes,
        frequencies,
        1,
    )

    np.testing.assert_allclose(peak_frequency, 1.25, atol=1e-12)
    np.testing.assert_allclose(peak_value, 0.0, atol=1e-12)


def test_compute_rms_returns_linear_amplitude():
    fs = 16000
    time = np.arange(fs, dtype=float) / fs
    signal = np.sin(2.0 * np.pi * 200.0 * time)

    rms = compute_rms(
        signal,
        fs=fs,
        frameshift_ms=5.0,
        F0=np.zeros(200, dtype=float),
        window_ms=20.0,
    )

    np.testing.assert_allclose(
        rms[10:-10],
        1.0 / np.sqrt(2.0),
        rtol=0,
        atol=1e-12,
    )


def test_smoothing_preserves_gaps_and_does_not_cross_segments():
    values = np.array([1.0, 3.0, np.nan, 100.0, 104.0], dtype=float)

    smoothed = smooth_preserving_gaps(values, window_size=3)

    np.testing.assert_allclose(
        smoothed,
        [2.0, 2.0, np.nan, 102.0, 102.0],
        equal_nan=True,
    )
