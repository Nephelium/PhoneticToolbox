"""P11 Linux trusted-host process primitive using transient user systemd units.

This low-level API accepts host-owned argv only (like native.windows.OwnedProcess).
It is not an HTTP/worker command endpoint. Callers must select a fixed application
entry. Resource limits are installed by systemd before the target is executed.
No shell, sudo, persistent unit, system setting, or RLIMIT-only fallback is used.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time
from uuid import uuid4

from ..io.limits import Cancelled, FormatError, LimitError


def _show(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit, '--no-pager',
                             '--property=LoadState,ActiveState,ControlGroup,Result,MemoryPeak,ExecMainStatus,MainPID'],
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=3)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def _sample(unit, evidence):
    state = _show(unit)
    if state.get('Result'):
        evidence['systemd_result'] = state['Result']
    peak = state.get('MemoryPeak', '')
    if peak.isdigit() and int(peak) < (1 << 63):
        evidence['memory_peak_bytes'] = max(evidence.get('memory_peak_bytes', 0), int(peak))
    group = state.get('ControlGroup', '')
    if group and group.endswith('/' + unit) and '..' not in Path(group).parts:
        path = Path('/sys/fs/cgroup') / group.lstrip('/')
        evidence['cgroup_path'] = str(path)
        for name in ('memory.max', 'memory.swap.max', 'cpu.max', 'pids.max', 'memory.events',
                     'memory.peak', 'memory.current', 'memory.pressure', 'cpu.stat', 'pids.current'):
            try:
                evidence[name] = (path/name).read_text(encoding='ascii').strip()
            except OSError:
                pass
        group_peak = evidence.get('memory.peak', '')
        if group_peak.isdigit():
            evidence['memory_peak_bytes'] = max(evidence.get('memory_peak_bytes', 0), int(group_peak))
    return state


def _cleanup(unit, process, evidence):
    """Only this unpredictable unit and our own systemd-run client may be stopped."""
    try:
        _sample(unit, evidence)
        # A cancellation can race StartTransientUnit. Keep the launch guard and
        # its lock alive until the client exits; then recheck the exact group.
        for _ in range(3):
            try:
                subprocess.run(['systemctl', '--user', 'stop', unit], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            except subprocess.TimeoutExpired:
                subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=all', '--signal=SIGKILL', unit],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=3)
            try:
                process.wait(timeout=2)
                break
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            evidence['cleaned'] = False
            # Do not kill a launch guard with an in-flight service start. Its
            # inherited lock keeps all subsequent callers out until recovery.
            raise RuntimeError('linux_process_group_cleanup_failed')
        state = _sample(unit, evidence)
        group = evidence.get('cgroup_path')
        populated = False
        if group:
            events = Path(group) / 'cgroup.events'
            try:
                populated = 'populated 1' in events.read_text(encoding='ascii')
            except FileNotFoundError:
                pass
        inactive = (state.get('LoadState') == 'not-found' or
                    state.get('LoadState') == 'loaded' and state.get('ActiveState') in ('inactive', 'failed'))
        evidence['cleaned'] = not populated and inactive and process.poll() is not None
        if not evidence['cleaned']:
            raise RuntimeError('linux_process_group_cleanup_failed')
        # Failed transient units otherwise remain in the user's manager. Never reset all units.
        subprocess.run(['systemctl', '--user', 'reset-failed', unit], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
    finally:
        process.stdin.close()
        process.stdout.close()


def recover_abandoned_unit(unit):
    """Retire only an exact journal-owned unit before granting the next slot."""
    from .admission import UNIT
    if not UNIT.fullmatch(unit):
        raise FormatError('resource_cleanup_required')
    if _show(unit).get('LoadState') != 'not-found':
        subprocess.run(['systemctl', '--user', 'stop', unit], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8, check=True)
        observation = {}
        state = _sample(unit, observation)
        group = observation.get('cgroup_path')
        if group:
            events = Path(group)/'cgroup.events'
            if events.exists() and 'populated 1' in events.read_text():
                raise FormatError('resource_cleanup_required')
        if state.get('LoadState') != 'not-found' and state.get('ActiveState') not in ('inactive', 'failed'):
            raise FormatError('resource_cleanup_required')
        subprocess.run(['systemctl', '--user', 'reset-failed', unit], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)


def run_bounded(argv, payload, cwd, limits, *, stop=lambda: False, evidence=None, on_started=None, on_chunk=None):
    from ..resource_profiles import selected_profile
    if not isinstance(payload, bytes) or len(payload) > limits.input_bytes:
        raise LimitError('input_bytes_exceeded')
    if stop():
        raise Cancelled('cancelled')
    if sys.platform != 'linux' or not shutil.which('systemd-run') or not shutil.which('systemctl'):
        raise FormatError('linux_process_boundary_unavailable')
    evidence = evidence if evidence is not None else {}
    profile = selected_profile()
    unit = 'ptb-p11-' + uuid4().hex + '.service'
    limits = profile.limits(limits)
    if profile.shared_admission:
        from .admission import Admission
        with Admission(queue_limit=profile.queue_limit, queue_seconds=profile.queue_seconds,
                       recover=recover_abandoned_unit).acquire(unit, stop=stop, evidence=evidence) as descriptor:
            return _run_bounded(argv, payload, cwd, limits, stop=stop, evidence=evidence,
                                on_started=on_started, on_chunk=on_chunk, unit=unit, descriptor=descriptor)
    evidence['admission_profile'] = profile.name
    return _run_bounded(argv, payload, cwd, limits, stop=stop, evidence=evidence,
                        on_started=on_started, on_chunk=on_chunk, unit=unit)


def _run_bounded(argv, payload, cwd, limits, *, stop=lambda: False, evidence=None, on_started=None,
                 on_chunk=None, unit, descriptor=None):
    """Bound input, output, wall time, memory of all descendants, and CPU/pids.

    ``evidence`` receives only this unit's resource and cleanup observations even
    on failure. A missing user systemd manager is an explicit unavailable error.
    The returned bytes are engineering output, not a scientific verification.
    """
    if not isinstance(payload, bytes) or len(payload) > limits.input_bytes:
        raise LimitError('input_bytes_exceeded')
    if stop():
        raise Cancelled('cancelled')
    if sys.platform != 'linux' or not shutil.which('systemd-run') or not shutil.which('systemctl'):
        raise FormatError('linux_process_boundary_unavailable')
    if not argv or not Path(argv[0]).is_absolute() or not Path(argv[0]).is_file():
        raise ValueError('Explicit trusted executable required')
    cwd = Path(cwd).resolve(strict=True)
    if not cwd.is_dir():
        raise ValueError('Host-owned working directory required')
    evidence = evidence if evidence is not None else {}
    from ..resource_profiles import selected_profile
    profile = selected_profile()
    state = _show(unit)
    if state.get('LoadState') != 'not-found':
        raise FormatError('linux_process_boundary_unavailable')
    evidence.update(unit=unit, configured_memory_bytes=limits.process_bytes, cleaned=False)
    args = ['systemd-run', '--user', '--quiet', '--pipe', '--wait', '--unit=' + unit,
            '--working-directory=' + str(cwd), '--property=Type=exec',
            '--property=MemoryAccounting=yes', '--property=MemoryMax=' + str(limits.process_bytes),
            '--property=MemorySwapMax=0', '--property=TasksMax=64',
            '--property=KillMode=control-group', '--property=OOMPolicy=kill',
            '--property=TimeoutStopSec=3', '--property=RuntimeMaxSec=' + str(math.ceil(limits.timeout_seconds) + 2),
            '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1',
            '--setenv=PTB_RESOURCE_PROFILE='+profile.name]
    if profile.name == 'server-small':
        args.append('--property=CPUQuota=100%')
    args += ['--'] + [str(item) for item in argv]
    started = time.monotonic()
    if descriptor is not None:
        args = [sys.executable, '-I', '-B', str(Path(__file__).with_name('launch_guard.py')),
                str(descriptor), *args]
    process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               pass_fds=() if descriptor is None else (descriptor,))
    output = bytearray()
    received = 0
    offset = 0
    last_sample = -float('inf')
    notified = False
    selector = selectors.DefaultSelector()
    os.set_blocking(process.stdout.fileno(), False)
    os.set_blocking(process.stdin.fileno(), False)
    selector.register(process.stdout, selectors.EVENT_READ)
    if payload:
        selector.register(process.stdin, selectors.EVENT_WRITE)
    else:
        process.stdin.close()
    try:
        while selector.get_map() or process.poll() is None:
            if stop():
                raise Cancelled('cancelled')
            if time.monotonic() - started > limits.timeout_seconds:
                raise LimitError('native_timeout')
            if time.monotonic() - last_sample >= .1:
                state = _sample(unit, evidence)
                pid = state.get('MainPID', '')
                if not notified and pid.isdigit() and int(pid) > 0:
                    notified = True
                    evidence['main_pid'] = int(pid)
                    if on_started:
                        on_started(int(pid))
                last_sample = time.monotonic()
            for key, _ in selector.select(.02):
                if key.fileobj is process.stdin:
                    try:
                        count = os.write(process.stdin.fileno(), payload[offset:offset + 65536])
                        offset += count
                    except BrokenPipeError:
                        selector.unregister(process.stdin)
                        process.stdin.close()
                        continue
                    if offset == len(payload):
                        selector.unregister(process.stdin)
                        process.stdin.close()
                else:
                    chunk = os.read(process.stdout.fileno(), min(65536, limits.output_bytes - received + 1))
                    if not chunk:
                        selector.unregister(process.stdout)
                    else:
                        received += len(chunk)
                        if received > limits.output_bytes:
                            raise LimitError('output_bytes_exceeded')
                        if on_chunk: on_chunk(chunk)
                        else: output.extend(chunk)
        code = process.wait(timeout=3)
        _sample(unit, evidence)
        evidence['returncode'] = code
        if code != 0:
            if evidence.get('systemd_result') == 'oom-kill':
                raise LimitError('process_memory_exceeded')
            raise FormatError('native_exit_' + str(code))
        if offset != len(payload):
            raise FormatError('incomplete_process_input')
        return bytes(output)
    finally:
        selector.close()
        evidence['elapsed_seconds'] = time.monotonic() - started
        _cleanup(unit, process, evidence)
