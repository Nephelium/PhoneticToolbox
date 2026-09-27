"""Real systemd/cgroup tests, explicitly enabled on an authorized Linux host."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ptb_worker.io.limits import Cancelled, FormatError, LimitError, Limits
from ptb_worker.native.posix import run_bounded

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'tests/support/p11_process_fixture.py'
REAL = pytest.mark.skipif(sys.platform != 'linux' or os.environ.get('PTB_P11_SYSTEMD_TESTS') != '1',
                          reason='requires explicitly authorized real Linux user systemd')


def invoke(action, folder, *, payload=b'', limits=None, stop=lambda: False, evidence=None):
    evidence = evidence if evidence is not None else {}
    try:
        return run_bounded([sys.executable, str(FIXTURE), action, str(folder/'child.pid')],
                           payload, folder, limits or Limits(timeout_seconds=3), stop=stop,
                           evidence=evidence)
    finally:
        directory = os.environ.get('PTB_P11_EVIDENCE_DIR')
        if directory and evidence.get('unit'):
            path = Path(directory) / (evidence['unit'] + '.json')
            with path.open('x', encoding='utf-8') as stream:
                json.dump(dict(action=action, **evidence), stream, indent=2)


def gone(pid):
    path = Path('/proc') / str(pid) / 'stat'
    if not path.exists():
        return True
    # A reaped-later zombie consumes no task resources and cannot execute.
    return path.read_text().split(') ', 1)[1].split()[0] == 'Z'


@REAL
def test_real_echo_and_limits_before_child(tmp_path):
    evidence = {}
    result = invoke('echo', tmp_path, payload='中文 ə ɕ'.encode(), evidence=evidence)
    assert result == '中文 ə ɕ'.encode()
    assert evidence['cleaned'] is True
    assert evidence['configured_memory_bytes'] == 512_000_000
    assert evidence['unit'].startswith('ptb-p11-')


@REAL
def test_limits_observed_inside_actual_child_before_heavy_imports(tmp_path):
    values = json.loads(invoke('limits', tmp_path))
    assert values == {'memory.max': '512000000', 'memory.swap.max': '0',
                      'cpu.max': '100000 100000', 'pids.max': '64'}


@REAL
@pytest.mark.parametrize('action,error', [('crash', FormatError), ('descendant', LimitError), ('memory', LimitError)])
def test_failure_timeout_and_aggregate_memory_leave_no_child(tmp_path, action, error):
    evidence = {}
    limits = Limits(timeout_seconds=3, process_bytes=72_000_000 if action == 'memory' else 256_000_000)
    with pytest.raises(error):
        invoke(action, tmp_path, limits=limits, evidence=evidence)
    assert evidence['cleaned'] is True
    pid = int((tmp_path/'child.pid').read_text())
    assert gone(pid)
    if action == 'memory':
        assert evidence.get('systemd_result') == 'oom-kill'


@REAL
def test_cancel_does_not_kill_unrelated_owned_sentinel(tmp_path):
    sentinel = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    evidence = {}
    try:
        with pytest.raises(Cancelled):
            invoke('descendant', tmp_path, stop=lambda: (tmp_path/'child.pid').exists(), evidence=evidence)
        assert evidence['cleaned'] is True
        assert gone(int((tmp_path/'child.pid').read_text()))
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


@REAL
def test_output_budget_and_next_task_recovery(tmp_path):
    evidence = {}
    with pytest.raises(LimitError):
        invoke('output', tmp_path, limits=Limits(output_bytes=1000), evidence=evidence)
    assert evidence['cleaned'] is True
    assert invoke('echo', tmp_path, payload=b'recovered') == b'recovered'


def test_input_limit_rejected_before_platform_start(tmp_path):
    with pytest.raises(LimitError):
        invoke('echo', tmp_path, payload=b'xx', limits=Limits(input_bytes=1))
