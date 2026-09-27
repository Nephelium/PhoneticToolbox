"""Lifecycle invariants; simulated manager timing, never actual-server evidence."""
import io
import subprocess

import pytest

from ptb_worker.native import posix


class Client:
    def __init__(self, waits):
        self.waits=waits
        self.stdin=io.BytesIO(); self.stdout=io.BytesIO()
    def wait(self,timeout):
        self.waits-=1
        if self.waits>0: raise subprocess.TimeoutExpired('owned-client',timeout)
        return 0
    def poll(self): return None if self.waits>0 else 0


def test_cleanup_retries_late_start_before_releasing_slot(monkeypatch):
    calls=[]; client=Client(3); evidence={}
    monkeypatch.setattr(posix.subprocess,'run',lambda argv,**kwargs:calls.append(argv))
    monkeypatch.setattr(posix,'_sample',lambda unit,evidence:dict(LoadState='not-found'))
    posix._cleanup('ptb-p11-owned.service',client,evidence)
    assert len([a for a in calls if 'stop' in a])==3
    assert evidence['cleaned'] is True and client.poll()==0


def test_uncertain_launch_is_not_killed_or_marked_clean(monkeypatch):
    calls=[]; client=Client(20); evidence={}
    monkeypatch.setattr(posix.subprocess,'run',lambda argv,**kwargs:calls.append(argv))
    monkeypatch.setattr(posix,'_sample',lambda unit,evidence:dict(LoadState='not-found'))
    with pytest.raises(RuntimeError,match='cleanup_failed'):
        posix._cleanup('ptb-p11-owned.service',client,evidence)
    assert evidence['cleaned'] is False and client.poll() is None
    assert not any('reset-failed' in a for a in calls)
