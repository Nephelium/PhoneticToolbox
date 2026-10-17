"""Trusted host-only digest metadata shared across short Linux child processes."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat

from ..io.limits import FormatError


def _checked(path, directory=False):
    info=path.lstat()
    kind=stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not kind or info.st_uid!=os.getuid() or info.st_mode & 0o077 or (not directory and info.st_nlink!=1):
        raise FormatError('linux_runtime_cache_unsafe')


@contextmanager
def shared_digests(memory, *, root=None):
    """Never accept a cache directory or digest from an HTTP request.

    This is a same-UID trusted-host cache, not an authority independent of the
    host. Every hit must still match the runtime file's current stat signature
    and the current profile's expected SHA256. Changed files are fully rehashed.
    """
    if os.name!='posix':
        yield;return
    import fcntl
    parent=Path('/run/user',str(os.getuid())) if root is None else Path(root)
    if not parent.exists():
        yield;return
    _checked(parent,directory=True)
    folder=parent/'ptb-runtime-digests-v1';folder.mkdir(mode=0o700,exist_ok=True);_checked(folder,directory=True)
    lock_path=folder/'lock'
    fd=os.open(lock_path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'rb') as lock:
        _checked(lock_path);fcntl.flock(lock,fcntl.LOCK_EX)
        path=folder/'digests.json';loaded={}
        if path.exists() or path.is_symlink():
            _checked(path)
            try:
                if path.stat().st_size<=2_000_000:
                    data=json.loads(path.read_text('utf-8'))
                    if isinstance(data,dict) and len(data)<=4096:
                        for name,entry in data.items():
                            if (isinstance(name,str) and Path(name).is_absolute() and isinstance(entry,list) and len(entry)==2
                                and isinstance(entry[0],list) and len(entry[0])==6 and all(type(v) is int for v in entry[0])
                                and isinstance(entry[1],str) and len(entry[1])==64 and all(c in '0123456789abcdef' for c in entry[1])):
                                loaded[name]=(tuple(entry[0]),entry[1])
            except (OSError,ValueError,TypeError):pass
        # Merge the process cache last; each entry is rechecked against stat.
        original=dict(loaded);loaded.update(memory);memory.clear();memory.update(loaded)
        try:yield
        finally:
            if memory!=original:
                if len(memory)>4096:memory.clear()
                raw=json.dumps(memory,separators=(',',':')).encode()
                if len(raw)>2_000_000:raw=b'{}'
                temporary=folder/'digests.next'
                fd=os.open(temporary,os.O_CREAT|os.O_WRONLY|os.O_TRUNC|os.O_NOFOLLOW,0o600)
                with os.fdopen(fd,'wb') as stream:
                    _checked(temporary);stream.write(raw)
                os.replace(temporary,path)
