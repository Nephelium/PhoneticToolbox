"""One host-user execution lane shared by all Linux API/worker processes.

No database or daemon. flock protects a bounded FIFO journal; another flock is
inherited by our systemd-run client so a killed API cannot release the lane while
its child group is still running. An abandoned unit is cleaned before reuse.
All participating services must run under the same dedicated Unix account.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import stat
import time
from uuid import uuid4

from ..io.limits import Cancelled, FormatError, LimitError

UNIT = re.compile(r'^ptb-p11-[0-9a-f]{32}\.service$')


def process_identity(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def default_root():
    # Deliberately independent of runtime/cache/store paths and XDG overrides.
    return Path('/run/user', str(os.getuid()), 'ptb-resource-admission-v1')


def _open(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        os.close(fd)
        raise FormatError('resource_admission_unsafe')
    return os.fdopen(fd, 'r+b', buffering=0)


class Admission:
    def __init__(self, *, root=None, queue_limit=128, queue_seconds=300, recover=None):
        self.root = Path(root) if root is not None else default_root()
        self.root.mkdir(mode=0o700, exist_ok=True)
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise FormatError('resource_admission_unsafe')
        self.queue_limit, self.queue_seconds, self.recover = queue_limit, queue_seconds, recover

    @contextmanager
    def journal(self):
        import fcntl
        with _open(self.root/'journal.lock') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                with _open(self.root/'state.json') as file:
                    raw = file.read(131073)
                if len(raw) > 131072:
                    raise FormatError('resource_admission_corrupt')
                try:
                    state = json.loads(raw) if raw else dict(queue=[], active=None)
                    if not isinstance(state['queue'], list) or len(state['queue']) > self.queue_limit:
                        raise ValueError()
                except (ValueError, KeyError, TypeError):
                    raise FormatError('resource_admission_corrupt') from None
                yield state
                with _open(self.root/'state.next') as file:
                    file.truncate(0)
                    file.write(json.dumps(state, separators=(',', ':')).encode())
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(self.root/'state.next', self.root/'state.json')
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @contextmanager
    def acquire(self, unit, *, stop=lambda: False, evidence=None):
        import fcntl
        if not UNIT.fullmatch(unit):
            raise ValueError('Owned unit required')
        evidence = evidence if evidence is not None else {}
        evidence['cleaned'] = False
        started = time.monotonic()
        ticket = dict(id=uuid4().hex, pid=os.getpid(), start=process_identity(os.getpid()), unit=unit)
        admitted = False
        with _open(self.root/'active.lock') as lane:
            try:
                with self.journal() as state:
                    state['queue'] = [x for x in state['queue'] if process_identity(x['pid']) == x['start']]
                    if len(state['queue']) >= self.queue_limit:
                        raise LimitError('resource_queue_full')
                    state['queue'].append(ticket)
                    evidence['queue_depth_on_arrival'] = len(state['queue'])
                while not admitted:
                    if stop():
                        raise Cancelled('cancelled')
                    if time.monotonic()-started >= self.queue_seconds:
                        raise LimitError('resource_queue_timeout')
                    with self.journal() as state:
                        state['queue'] = [x for x in state['queue'] if process_identity(x['pid']) == x['start']]
                        if state['queue'] and state['queue'][0]['id'] == ticket['id']:
                            try:
                                fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                pass
                            else:
                                previous = state['active']
                                if previous:
                                    if not self.recover or not UNIT.fullmatch(previous['unit']):
                                        raise FormatError('resource_cleanup_required')
                                    self.recover(previous['unit'])
                                    evidence['recovered_previous_unit'] = previous['unit']
                                state['queue'].pop(0)
                                state['active'] = ticket
                                admitted = True
                    if not admitted:
                        time.sleep(.05)
                evidence['queue_wait_seconds'] = time.monotonic()-started
                evidence['admission_profile'] = 'server-small'
                # Caller MUST pass this descriptor to its owned launch guard.
                yield lane.fileno()
            finally:
                with self.journal() as state:
                    state['queue'] = [x for x in state['queue'] if x['id'] != ticket['id']]
                    # On cleanup failure retain the exact unit for recovery. Do
                    # not admit another group merely because the parent returned.
                    if admitted and evidence.get('cleaned') is True and state['active'] == ticket:
                        state['active'] = None
                # Closing only: LOCK_UN would release a lock inherited by a live
                # systemd-run client after a Python exception in its parent.
