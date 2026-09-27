"""Local configuration only; never accepts remote executable or file paths."""
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
import json
import os
import stat


class NodeError(Exception):
    """Only fixed, non-sensitive codes may cross the CLI/log boundary."""


def origin(value):
    try:
        u = urlsplit(value)
        if (u.scheme != 'https' or not u.hostname or u.username or u.password
                or u.path not in ('', '/') or u.query or u.fragment
                or any(c.isspace() for c in value)):
            raise ValueError()
        port = u.port or 443
        host = '[' + u.hostname + ']' if ':' in u.hostname else u.hostname
        return 'https://' + host + (':' + str(port) if port != 443 else '')
    except (ValueError, TypeError):
        raise NodeError('origin_invalid') from None


@dataclass(frozen=True)
class Config:
    server_origin: str
    state_dir: Path
    credential_file: Path
    cpu_percent: int = 50
    memory_bytes: int = 536870912
    parallelism: int = 1
    temporary_bytes: int = 1073741824
    reserve_disk_bytes: int = 1073741824
    work_hours: tuple[int, int] = (0, 24)

    @classmethod
    def load(cls, path):
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
            if data.pop('schema') != 'ptb-node-config/1':
                raise ValueError()
            data['server_origin'] = origin(data['server_origin'])
            for key in ('state_dir', 'credential_file'):
                data[key] = Path(data[key])
                if not data[key].is_absolute():
                    raise ValueError()
            data['work_hours'] = tuple(data.get('work_hours', (0, 24)))
            cfg = cls(**data)
            for key in ('cpu_percent', 'memory_bytes', 'parallelism', 'temporary_bytes', 'reserve_disk_bytes'):
                value = getattr(cfg, key)
                if type(value) is not int or value <= 0:
                    raise ValueError()
            if (not 1 <= cfg.cpu_percent <= 100 or cfg.parallelism != 1
                    or cfg.memory_bytes < 67108864 or len(cfg.work_hours) != 2
                    or any(type(x) is not int for x in cfg.work_hours)
                    or not 0 <= cfg.work_hours[0] < cfg.work_hours[1] <= 24):
                raise ValueError()
            return cfg
        except (OSError, KeyError, ValueError, TypeError):
            raise NodeError('config_invalid') from None

    def in_hours(self, local_time):
        return self.work_hours[0] <= local_time.hour < self.work_hours[1]


def credential(path):
    """No command-line secrets. Native Linux mode/owner required; no symlink."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o077 or info.st_nlink != 1):
                raise NodeError('credential_permissions')
            value = stream.read(8193)
        token = value.decode('ascii').strip()
        if not token or len(value) > 8192 or any(ord(c) <= 32 or ord(c) >= 127 for c in token):
            raise NodeError('credential_invalid')
        return token
    except (OSError, UnicodeError):
        raise NodeError('credential_unavailable') from None


def save_credential(path, value):
    """Explicit local provisioning, not node enrollment or remote revocation."""
    from .storage import private_dir
    path = Path(path)
    private_dir(path.parent)
    if (not value or len(value) > 8192 or any(ord(c) <= 32 or ord(c) >= 127 for c in value)):
        raise NodeError('credential_invalid')
    # No silent overwrite/rotation. Administrator must provide a distinct file
    # and explicitly select it in local config for an existing identity.
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w', encoding='ascii') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        raise NodeError('credential_write_failed') from None
