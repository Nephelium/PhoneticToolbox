"""Opt-in actual-server tests for the shared gate plus systemd launch lifecycle."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark=pytest.mark.skipif(sys.platform!='linux' or os.environ.get('PTB_P11_SYSTEMD_TESTS')!='1',
                            reason='requires authorized actual Linux user systemd')

CHILD='''
import json,os,sys
from pathlib import Path
from ptb_worker.native.posix import run_bounded
from ptb_worker.io.limits import Limits
folder=Path(sys.argv[1]); evidence={}
def started(pid):
    if sys.argv[2]=='crash': os._exit(7)
try:
    raw=run_bounded([sys.executable,'-c',
        'import time,json; start=time.monotonic(); time.sleep(.6); print(json.dumps([start,time.monotonic()]))'],
        b'',folder,Limits(process_bytes=128_000_000,timeout_seconds=5),evidence=evidence,on_started=started)
    value=dict(interval=json.loads(raw),process=evidence)
finally:
    (folder/'result.json').write_text(json.dumps(value))
'''


def test_independent_api_processes_cannot_overlap_groups(tmp_path):
    children=[]
    try:
        for index in range(4):
            folder=tmp_path/str(index); folder.mkdir()
            children.append(subprocess.Popen([sys.executable,'-c',CHILD,str(folder),'normal']))
        for child in children: assert child.wait(timeout=30)==0
        values=[json.loads((tmp_path/str(index)/'result.json').read_text()) for index in range(4)]
        ordered=sorted(v['interval'] for v in values)
        assert all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:]))
        assert all(v['process']['cleaned'] for v in values)
        assert all(v['process']['admission_profile']=='server-small' for v in values)
        assert max(v['process']['queue_wait_seconds'] for v in values)>.6
    finally:
        for child in children:
            if child.poll() is None: child.terminate(); child.wait(timeout=10)


def test_api_crash_cannot_release_running_group(tmp_path):
    crashed=tmp_path/'crash'; crashed.mkdir()
    parent=subprocess.Popen([sys.executable,'-c',CHILD,str(crashed),'crash'])
    assert parent.wait(timeout=15)==7
    recovered=tmp_path/'recovery'; recovered.mkdir()
    successor=subprocess.Popen([sys.executable,'-c',CHILD,str(recovered),'normal'])
    assert successor.wait(timeout=20)==0
    evidence=json.loads((recovered/'result.json').read_text())['process']
    assert evidence['cleaned'] and evidence['recovered_previous_unit']
    assert evidence['queue_wait_seconds']>.1
