import pytest
from phonetic_core.models.associations import Interval, Tier
from phonetic_core.segmentation import plan_segments


@pytest.mark.parametrize('tail', ['', 'sil', '<eps>'])
def test_unused_silent_tail_past_audio_does_not_reject_labelled_segments(tail):
    tiers = (Tier('区域', (Interval(0, .25, ''), Interval(.25, .75, 'NP1'),
                         Interval(.75, 10, tail))),)
    result = plan_segments(tiers, '区域', 8000, 8000)
    assert [(s.label, s.first, s.last) for s in result] == [('NP1', 2000, 6000)]


@pytest.mark.parametrize('intervals', [
    (Interval(0, 2, 'NP1'),),
    (Interval(0, .8, ''), Interval(.7, .9, 'NP1')),
    (Interval(0, .9, 'NP1'), Interval(.8, 10, '')),
    (Interval(-1, 0, ''), Interval(0, .5, 'NP1')),
])
def test_labelled_out_of_range_and_invalid_silent_intervals_still_rejected(intervals):
    with pytest.raises(ValueError):
        plan_segments((Tier('区域', intervals),), '区域', 8000, 8000)
