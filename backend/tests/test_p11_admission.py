"""P11-PERF actual cross-process locks. cgroup tests live in test_p11_posix."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import pytest

from ptb_worker.io.limits import Cancelled, FormatError, LimitError, Limits
from ptb_worker.resource_profiles import selected_profile

LINUX = pytest.mark.skipif(sys.platform != 'linux', reason='real Linux flock and /proc required')


def unit():
    return 'ptb-p11-' + uuid4().hex + '.service'


def test_desktop_preserves_module_limits(monkeypatch):
    monkeypatch.setenv('PTB_RESOURCE_PROFILE', 'desktop-local')
    limits = Limits(process_bytes=3_000_000_000)
    assert selected_profile().limits(limits) is limits


@pytest.mark.parametrize('name', ['trusted-worker', 'misspelled'])
def test_unavailable_profiles_fail_closed(monkeypatch, name):
    monkeypatch.setenv('PTB_RESOURCE_PROFILE', name)
    with pytest.raises(FormatError):
        selected_profile()


def test_server_budget_does_not_change_scientific_limits(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setenv('PTB_RESOURCE_PROFILE', 'server-small')
    old = Limits(process_bytes=3_000_000_000)
    new = selected_profile().limits(old)
    assert new.process_bytes == 1_073_741_824
    assert {k:v for k,v in vars(new).items() if k != 'process_bytes'} == {
        k:v for k,v in vars(old).items() if k != 'process_bytes'}


CHILD = '''
import json, sys, time
from pathlib import Path
from ptb_worker.native.admission import Admission
root, name, label = sys.argv[1:]
evidence = {}
with Admission(root=root).acquire(name, evidence=evidence):
    with open(Path(root)/'events', 'a') as f: f.write(label+' start\\n')
    time.sleep(.1)
    with open(Path(root)/'events', 'a') as f: f.write(label+' end\\n')
    evidence['cleaned'] = True
'''


def wait_queue(root, count):
    deadline = time.monotonic()+10
    while time.monotonic() < deadline:
        if len(json.loads((root/'state.json').read_text())['queue']) == count:
            return
        time.sleep(.02)
    raise AssertionError('queue did not reach expected size')


@LINUX
def test_real_cross_process_fifo_no_overlap(tmp_path):
    from ptb_worker.native.admission import Admission
    gate = Admission(root=tmp_path)
    evidence = {}
    children = []
    try:
        with gate.acquire(unit(), evidence=evidence):
            for index in range(4):
                children.append(subprocess.Popen([sys.executable, '-c', CHILD, str(tmp_path), unit(), str(index)]))
                wait_queue(tmp_path, index+1)
            evidence['cleaned'] = True
        for child in children:
            assert child.wait(timeout=15) == 0
        assert (tmp_path/'events').read_text().splitlines() == [f'{i} {event}' for i in range(4) for event in ('start','end')]
        assert json.loads((tmp_path/'state.json').read_text()) == {'queue': [], 'active': None}
    finally:
        for child in children:
            if child.poll() is None: child.terminate(); child.wait(timeout=5)


@LINUX
def test_wait_cancel_timeout_and_recovery(tmp_path):
    from ptb_worker.native.admission import Admission
    evidence = {}
    with Admission(root=tmp_path).acquire(unit(), evidence=evidence):
        for stop, error in [(lambda: True, Cancelled), (lambda: False, LimitError)]:
            with pytest.raises(error):
                with Admission(root=tmp_path, queue_seconds=.1).acquire(unit(), stop=stop):
                    pytest.fail('second lane opened')
        assert json.loads((tmp_path/'state.json').read_text())['queue'] == []
        evidence['cleaned'] = True
    with Admission(root=tmp_path).acquire(unit(), evidence=evidence):
        evidence['cleaned'] = True


@LINUX
def test_cleanup_failure_blocks_and_exact_unit_recovers(tmp_path):
    from ptb_worker.native.admission import Admission
    previous = unit()
    with Admission(root=tmp_path).acquire(previous):
        pass  # Deliberate missing clean acknowledgement.
    with pytest.raises(FormatError, match='resource_cleanup_required'):
        with Admission(root=tmp_path).acquire(unit()):
            pytest.fail('unclean group admitted')
    recovered = []
    evidence = {}
    with Admission(root=tmp_path, recover=recovered.append).acquire(unit(), evidence=evidence):
        evidence['cleaned'] = True
    assert recovered == [previous]


@LINUX
def test_inherited_lock_survives_parent_crash(tmp_path):
    from ptb_worker.native.admission import Admission
    code = '''
import os, subprocess, sys
from pathlib import Path
from ptb_worker.native.admission import Admission
with Admission(root=sys.argv[1]).acquire(sys.argv[2]) as fd:
    p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1.5)'], pass_fds=(fd,))
    Path(sys.argv[1], 'child.pid').write_text(str(p.pid))
    os._exit(7)
'''
    parent = subprocess.Popen([sys.executable, '-c', code, str(tmp_path), unit()])
    assert parent.wait(timeout=10) == 7
    with pytest.raises(LimitError, match='resource_queue_timeout'):
        with Admission(root=tmp_path, queue_seconds=.2, recover=lambda _: None).acquire(unit()):
            pytest.fail('parent crash released live inherited lane')
    evidence = {}
    with Admission(root=tmp_path, recover=lambda _: None).acquire(unit(), evidence=evidence):
        evidence['cleaned'] = True
    assert evidence['recovered_previous_unit']


@LINUX
def test_launch_guard_retains_lock_even_if_client_closes_descriptors(tmp_path):
    from ptb_worker.native.admission import Admission
    from ptb_worker.native import launch_guard
    client=tmp_path/'systemd-run'
    client.write_text('#!'+sys.executable+'\nimport os,time\nos.closerange(3,1024)\ntime.sleep(1.5)\n')
    client.chmod(0o700)
    code='''
import os, subprocess, sys
from ptb_worker.native.admission import Admission
with Admission(root=sys.argv[1]).acquire(sys.argv[2]) as fd:
    subprocess.Popen([sys.executable,'-I',sys.argv[3],str(fd),'systemd-run'],pass_fds=(fd,))
    os._exit(7)
'''
    parent=subprocess.Popen([sys.executable,'-c',code,str(tmp_path),unit(),launch_guard.__file__],
                            env={**os.environ,'PATH':str(tmp_path)+os.pathsep+os.environ['PATH']})
    assert parent.wait(timeout=10)==7
    with pytest.raises(LimitError,match='resource_queue_timeout'):
        with Admission(root=tmp_path,queue_seconds=.2,recover=lambda _:None).acquire(unit()):
            pytest.fail('closed client descriptors released guard lock')
    evidence={}
    with Admission(root=tmp_path,recover=lambda _:None).acquire(unit(),evidence=evidence):
        evidence['cleaned']=True


@LINUX
def test_corrupt_journal_fail_closed(tmp_path):
    from ptb_worker.native.admission import Admission
    gate = Admission(root=tmp_path)
    (tmp_path/'state.json').write_text('broken')
    (tmp_path/'state.json').chmod(0o600)
    with pytest.raises(FormatError, match='resource_admission_corrupt'):
        with gate.acquire(unit()):
            pytest.fail('corrupt state admitted')


@LINUX
def test_queue_full_and_dead_waiter_recovery(tmp_path):
    from ptb_worker.native.admission import Admission
    evidence = {}
    waiter = None
    try:
        with Admission(root=tmp_path).acquire(unit(), evidence=evidence):
            waiter = subprocess.Popen([sys.executable, '-c', CHILD, str(tmp_path), unit(), 'dead'])
            wait_queue(tmp_path, 1)
            with pytest.raises(LimitError, match='resource_queue_full'):
                with Admission(root=tmp_path, queue_limit=1).acquire(unit()):
                    pytest.fail('full queue admitted')
            waiter.kill(); waiter.wait(timeout=5)
            evidence['cleaned'] = True
        evidence = {}
        with Admission(root=tmp_path).acquire(unit(), evidence=evidence):
            evidence['cleaned'] = True
        assert json.loads((tmp_path/'state.json').read_text())['queue'] == []
    finally:
        if waiter is not None and waiter.poll() is None: waiter.terminate(); waiter.wait(timeout=5)
