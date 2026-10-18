"""FFT shape and unchanged LPC, independently checked against a direct DFT."""
import numpy as np
import pytest
from phonetic_core.lpc.spectrum import fft_reference_spectrum


def test_reference_matches_direct_dft_without_mutating_input():
    fs = 16000
    audio = np.array([.2, -.3, .7, .1, -.2, .8, .4])
    before = audio.copy()
    frequency, magnitude = fft_reference_spectrum(audio, fs)
    emphasized = np.r_[audio[0], audio[1:] - .97 * audio[:-1]]
    window = .54 - .46 * np.cos(2 * np.pi * np.arange(7) / 6)
    # Independent 8-point DFT, including correct DC/Nyquist factors.
    values = np.exp(-2j * np.pi * np.outer(np.arange(5), np.arange(7)) / 8) @ (emphasized * window)
    amplitude = abs(values) / window.sum() * np.array([1, 2, 2, 2, 1])
    np.testing.assert_array_equal(frequency, [0, 2000, 4000, 6000, 8000])
    np.testing.assert_allclose(magnitude, 20 * np.log10(amplitude + 1e-10), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(audio, before)


@pytest.mark.parametrize('length', [3, 4279, 48000])
def test_fft_covers_whole_selection_and_nyquist(length):
    audio = np.zeros(length); audio[-1] = 1
    frequency, magnitude = fft_reference_spectrum(audio, 44100)
    assert len(frequency) == len(magnitude) <= 32769
    assert frequency[0] == 0 and frequency[-1] == 22050
    assert np.isfinite(magnitude).all()


def test_silent_reference_is_finite():
    _, magnitude = fft_reference_spectrum(np.zeros(64), 16000)
    np.testing.assert_array_equal(magnitude, np.full(33, -200.))
