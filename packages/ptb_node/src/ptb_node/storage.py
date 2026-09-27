"""Private node state and disposable attempt storage; no external path cleanup."""
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import stat
from uuid import uuid4
from .config import NodeError


def private_dir(path):
    path = Path(path)
    if not path.is_absolute():
        raise NodeError('state_path_invalid')
    # Refuse symlink ancestors, including the final component, before creating anything.
    for item in reversed((path, *path.parents)):
        if item.is_symlink():
            raise NodeError('state_path_invalid')
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise NodeError('state_permissions')
    return path


def write_json(path, value):
    path = Path(path)
    temporary = path.parent / ('.write-' + uuid4().hex)
    try:
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'w') as stream:
            json.dump(value, stream, ensure_ascii=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class Instance:
    def __init__(self, state):
        self.state = private_dir(state)
        self.fd = None

    def __enter__(self):
        self.fd = os.open(self.state / 'instance.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
            raise NodeError('already_running') from None
        return self

    def __exit__(self, *args):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        # Never unlink the lock inode: other waiters might have it open.


class Attempts:
    NAME = re.compile(r'a-[0-9a-f]{32}\Z')

    def __init__(self, state, byte_limit, reserve_bytes):
        self.root = private_dir(Path(state) / 'attempts')
        self.byte_limit = byte_limit
        self.reserve_bytes = reserve_bytes

    def create(self, identity, expires_at, maximum_bytes):
        if (type(maximum_bytes) is not int or maximum_bytes <= 0
                or maximum_bytes > self.byte_limit
                or shutil.disk_usage(self.root).free < maximum_bytes + self.reserve_bytes):
            raise NodeError('disk_budget_unavailable')
        if any(self.root.iterdir()):
            raise NodeError('residual_cleanup_required')
        directory = self.root / ('a-' + uuid4().hex)
        directory.mkdir(mode=0o700)
        write_json(directory / 'owner.json', {'schema': 'ptb-node-attempt/1',
                   'attempt_id': identity.attempt_id, 'generation': identity.generation,
                   'expires_at': expires_at, 'maximum_bytes': maximum_bytes})
        return directory

    def open_new(self, directory):
        self._owned(directory)
        # Local unpredictable names only, with no server-selected suffix/path.
        path = directory / ('f-' + uuid4().hex)
        return path, os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb')

    def _owned(self, directory):
        directory = Path(directory)
        if directory.parent != self.root or not self.NAME.fullmatch(directory.name) or directory.is_symlink():
            raise NodeError('cleanup_not_owned')
        try:
            fd = os.open(directory / 'owner.json', os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'r') as stream:
                value = json.loads(stream.read(4097))
            if value.get('schema') != 'ptb-node-attempt/1':
                raise ValueError()
        except (OSError, ValueError):
            raise NodeError('cleanup_not_owned') from None

    def cleanup(self, directory):
        self._owned(directory)
        if not shutil.rmtree.avoids_symlink_attacks:
            raise NodeError('safe_cleanup_unavailable')
        try:
            shutil.rmtree(directory)
        except OSError:
            raise NodeError('cleanup_failed') from None

    def recover(self):
        # Never resume old attempts after process restart. Deleting ALL owned residuals
        # is more conservative than reusing unexpired data with a fresh generation.
        for item in self.root.iterdir():
            self.cleanup(item)


class BudgetWriter:
    def __init__(self, stream, maximum, reserve=0):
        self.stream, self.maximum, self.reserve, self.written = stream, maximum, reserve, 0

    def write(self, data):
        if self.written + len(data) > self.maximum:
            raise NodeError('disk_budget_exceeded')
        fs = os.fstatvfs(self.stream.fileno())
        if fs.f_bavail * fs.f_frsize < len(data) + self.reserve:
            raise NodeError('disk_budget_unavailable')
        count = self.stream.write(data)
        if count != len(data):
            raise NodeError('disk_write_failed')
        self.written += count
        return count
