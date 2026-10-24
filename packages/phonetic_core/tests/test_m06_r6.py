"""M06 review fixes: independent AV controls and sample-resolvable pauses."""
from copy import deepcopy
import numpy as np
import pytest
from phonetic_core.synthesis.klatt.api import defaults, generate, synthesize


def config(sequence, duration=.6):
    c = defaults()
    c['duration'] = duration
    c['sequence'] = sequence
    for curve in c['curves'].values():
        curve['points'][-1][0] = duration
    return c


@pytest.mark.parametrize('override', [None, 0., 70.])
def test_generation_retains_complete_independent_av_control(override):
    c = config('a a')
    c['curves']['AV'] = dict(points=[[0., 60.], [.125, 55.], [.6, 65.]], override=override)
    before = deepcopy(c)
    out = generate(c)
    assert c == before
    assert out['curves']['AV'] == before['curves']['AV']
    assert len(out['silence']) == 1


def test_removing_pause_does_not_leave_an_old_av_hole():
    c = generate(config('a a'))
    c['sequence'] = 'a'
    out = generate(c)
    fresh = generate(config('a'))
    assert out['silence'] == []
    assert out['curves']['AV'] == fresh['curves']['AV']
    np.random.seed(23)
    actual = synthesize(out)
    np.random.seed(23)
    expected = synthesize(fresh)
    np.testing.assert_array_equal(actual, expected)


def test_short_explicit_pause_survives_between_curve_grid_points():
    c = generate(config('a //////i+', .3))
    weights = [1., .5**6, 1.1]
    start = .3 * weights[0] / sum(weights)
    end = .3 * sum(weights[:2]) / sum(weights)
    np.testing.assert_allclose(c['silence'], [[start, end]], rtol=0, atol=1e-16)
    assert end - start < .01
    np.random.seed(23)
    audio = synthesize(c)
    i0, i1 = round(start * 16000), round(end * 16000)
    assert i1 - i0 == 35
    assert not np.any(audio[i0:i1])
    assert np.any(audio[:i0]) and np.any(audio[i1:])


@pytest.mark.parametrize('rate', [8000, 16000, 44100, 48000])
def test_multiple_pause_boundaries_do_not_drift_with_independent_rounding(rate):
    c = config('a //////i+ //////u-', .3)
    c['sample_rate'] = rate
    c['fade_in'] = c['fade_out'] = 0
    c = generate(c)
    np.random.seed(23)
    audio = synthesize(c)
    assert len(audio) == round(.3 * rate)
    for start, end in c['silence']:
        assert not np.any(audio[round(start * rate):round(end * rate)])
