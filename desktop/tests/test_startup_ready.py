import os
import subprocess
import sys
import pytest
from ptb_desktop.startup_ready import ENVIRONMENT,ReadyEvent,signal_ready

pytestmark=pytest.mark.skipif(os.name!='nt',reason='Windows named readiness event')


def test_readiness_is_isolated_and_signalled_by_another_process(monkeypatch):
    first,second=ReadyEvent(),ReadyEvent()
    try:
        assert not first.is_set() and not second.is_set()
        env=os.environ.copy();env[ENVIRONMENT]=first.name
        result=subprocess.run([sys.executable,'-B','-c',
            'from ptb_desktop.startup_ready import signal_ready;assert signal_ready()'],
            env=env,capture_output=True,text=True,timeout=10)
        assert result.returncode==0,result.stderr
        assert first.is_set() and not second.is_set()
        monkeypatch.setenv(ENVIRONMENT,second.name);assert signal_ready() and second.is_set()
    finally:first.close();second.close()


def test_missing_stale_or_unowned_event_does_not_create_a_signal(monkeypatch):
    for name in ('','arbitrary-system-event','Local\\PTBStartupReady-'+'0'*32):
        monkeypatch.setenv(ENVIRONMENT,name);assert not signal_ready()
