"""P16: requested backend policy, including cancellation before fallback."""
import numpy as np
import pytest
from phonetic_core.ports.acoustic import ReaperTrack
from ptb_worker.io.limits import Cancelled, LimitError
from ptb_worker.reaper_policy import PolicyReaper


def track(kind):return ReaperTrack(np.array([0.]),np.array([120.]),kind)


def test_python_only_never_constructs_native():
    def forbidden():raise AssertionError('native opened')
    assert PolicyReaper('python_only',forbidden,lambda *a,**kw:track('reaper_python'))().actual_backend=='reaper_python'


@pytest.mark.parametrize('during_init',[True,False])
def test_explicit_fallback_records_native_failure(during_init):
    def fail(*a,**kw):raise OSError('private path')
    adapter=PolicyReaper('native_then_python',fail if during_init else lambda:fail,lambda *a,**kw:track('reaper_python'))
    result=adapter()
    assert result.actual_backend=='reaper_python'
    assert result.reason=='native_failed'


def test_required_native_does_not_fallback():
    def fail():raise OSError('missing')
    def forbidden(*a,**kw):raise AssertionError('fallback executed')
    with pytest.raises(OSError):PolicyReaper('native_required',fail,forbidden)()


@pytest.mark.parametrize('error',[Cancelled('cancelled'),LimitError('native_timeout'),MemoryError()])
def test_resource_abort_never_falls_back(error):
    def fail():raise error
    def forbidden(*a,**kw):raise AssertionError('fallback executed')
    with pytest.raises(type(error)):PolicyReaper('native_then_python',fail,forbidden)()
