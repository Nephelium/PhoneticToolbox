"""M04 A03/A06/A11/A13/A14 explicit public API boundaries."""
import numpy as np
import pytest
from phonetic_core.lpc import (LPCConfig, LPCError, LPCCancelled, MAX_ROI_SAMPLES,
                               compute_spectrum, mono_samples, select_segment)


@pytest.mark.parametrize('kwargs', [dict(order=x) for x in (0, 201, 2.5, True, '50')] + [
    dict(freq_max_hz=x) for x in (0, 99, 48001, float('nan'), 10**400)] + [
    dict(amp_min_db=-201), dict(amp_max_db=101), dict(amp_min_db=35),
    dict(amp_max_db=float('inf')), dict(dynamic_y=1)])
def test_invalid_config(kwargs):
    with pytest.raises(LPCError) as error:
        LPCConfig(**kwargs)
    assert error.value.code == 'invalid_config'


@pytest.mark.parametrize('fs', [0, -1, True, '48000', 1.5, float('nan'), float('inf'), 768001])
def test_invalid_rate(fs):
    with pytest.raises(LPCError, match='采样率'):
        compute_spectrum(np.ones(100), fs)


@pytest.mark.parametrize('audio', [[], np.ones((100, 2)), np.array([1+2j]*100),
                                  ['1']*100, [True]*100, [float('nan')]*100,
                                  [float('inf')]*100, np.ones((2, 3, 4))])
def test_invalid_analysis_array(audio):
    with pytest.raises(LPCError):
        compute_spectrum(audio, 48000)


def test_long_roi_rejected_before_algorithm(monkeypatch):
    import phonetic_core.lpc.spectrum as module
    def unexpected(*args, **kwargs):
        pytest.fail('Over-budget input reached quadratic correlation')
    monkeypatch.setattr(module, 'compute_lpc_spectrum', unexpected)
    with pytest.raises(LPCError) as error:
        compute_spectrum(np.ones(MAX_ROI_SAMPLES+1), 48000)
    assert error.value.code == 'roi_too_large'


def test_cancellation_before_and_after_computation():
    audio = np.random.default_rng(4).normal(size=100)
    for answers in ((True,), (False, True)):
        calls = iter(answers)
        with pytest.raises(LPCCancelled):
            compute_spectrum(audio, 48000, cancelled=lambda: next(calls))


def test_short_and_singular_failures_are_distinct():
    for audio, code in [(np.ones(51), 'segment_too_short'), (np.zeros(100), 'solver_failed')]:
        with pytest.raises(LPCError) as error:
            compute_spectrum(audio, 48000)
        assert error.value.code == code


def test_roi_half_open_and_independent():
    original = np.arange(160, dtype=np.float64)
    segment = select_segment(original, 16000, .00109, .00509)
    np.testing.assert_array_equal(segment, np.arange(17, 81))
    segment[:] = 0
    assert original[17] == 17
    np.testing.assert_array_equal(select_segment(original, 16000, .005, .01), original[80:])


@pytest.mark.parametrize('start,end', [(-.1,.1), (.1,.1), (.2,.1), (0,2),
                                     (float('nan'),.1), (0,float('inf')), (0, .000001)])
def test_bad_roi(start, end):
    with pytest.raises(LPCError):
        select_segment(np.ones(16000), 16000, start, end)


@pytest.mark.parametrize('samples', [np.empty((2,0)), np.ones((2,2,2)), np.array(['a']),
                                    np.array([1j]), np.array([float('nan')])])
def test_bad_decoded_audio(samples):
    with pytest.raises(LPCError):
        mono_samples(samples)


def test_mono_copy_and_overflow():
    source = np.arange(100, dtype=np.float64)
    mono_samples(source)[:] = 0
    assert source[99] == 99
    with pytest.raises(LPCError):
        compute_spectrum(np.full(100, 1e308), 48000)


def test_display_limit_does_not_change_spectrum():
    audio = np.random.default_rng(5).normal(size=400)
    small = compute_spectrum(audio, 16000, LPCConfig(freq_max_hz=100))
    beyond_nyquist = compute_spectrum(audio, 16000, LPCConfig(freq_max_hz=48000))
    assert small.magnitude_db.tobytes() == beyond_nyquist.magnitude_db.tobytes()
    assert len(small.frequencies_hz) == 1024 and small.frequencies_hz[-1] < 8000
