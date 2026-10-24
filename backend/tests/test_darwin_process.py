"""Actual macOS process cleanup, transport budgets and cancellation."""
import os,sys,time
import pytest
from ptb_worker.io.limits import Limits,LimitError,Cancelled
from ptb_worker.native.darwin import run_bounded,group_members

pytestmark=pytest.mark.skipif(sys.platform!='darwin',reason='macOS native process boundary')

def test_roundtrip_bounded_pipe(tmp_path):
    evidence={}
    output=run_bounded([sys.executable,'-c','import sys;sys.stdout.buffer.write(sys.stdin.buffer.read())'],b'roundtrip',tmp_path,Limits(),evidence=evidence)
    assert output==b'roundtrip' and evidence['cleaned'] and evidence['memory_peak_bytes']>0

def test_wall_time_cleans_owned_group(tmp_path):
    evidence={}
    with pytest.raises(LimitError,match='native_timeout'):
        run_bounded([sys.executable,'-c','import time;time.sleep(10)'],b'',tmp_path,Limits(timeout_seconds=.2),evidence=evidence)
    assert evidence['cleaned']
    assert not group_members(evidence['main_pid'])

def test_output_budget_is_enforced_while_child_is_writing(tmp_path):
    evidence={}
    with pytest.raises(LimitError,match='output_bytes_exceeded'):
        run_bounded([sys.executable,'-c','import os;os.write(1,b"x"*1000000)'],b'',tmp_path,Limits(output_bytes=1024),evidence=evidence)
    assert evidence['cleaned']

def test_cancellation_stops_only_its_owned_child(tmp_path):
    started=time.monotonic();evidence={}
    with pytest.raises(Cancelled):
        run_bounded([sys.executable,'-c','import time;time.sleep(10)'],b'',tmp_path,Limits(),stop=lambda:time.monotonic()-started>.2,evidence=evidence)
    assert evidence['cleaned']

def test_sigterm_ignoring_descendant_is_killed(tmp_path):
    code='import subprocess,sys,time;subprocess.Popen([sys.executable,"-c","import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)"]);time.sleep(20)'
    evidence={}
    with pytest.raises(LimitError):run_bounded([sys.executable,'-c',code],b'',tmp_path,Limits(timeout_seconds=.4),evidence=evidence)
    assert not group_members(evidence['main_pid'])
