import numpy as np
import pytest
from phonetic_core.acoustic.jitter_shimmer import _extract_f0_for_wm


def empty_irapt(*args, **kwargs):
    return np.array([]), np.array([]), np.array([])


@pytest.mark.parametrize('samples,reason', [(np.zeros(0),'empty_audio'),(np.zeros(1),'praat_failed'),(np.zeros(16000),'praat_insufficient')])
def test_failed_wm_backend_is_not_reported_as_success(samples, reason):
    events=[]
    values,times=_extract_f0_for_wm(samples,16000,60.,880.,f0_provider=empty_irapt,backend_events=events)
    assert values.size == times.size == 0
    assert events == [{'stage':'wm_f0','actual':'unavailable','reason':reason}]
