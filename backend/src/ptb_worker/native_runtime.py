"""Process-owned ASCII entry points for native libraries with legacy paths.

Junctions refer to the already verified runtime without copying or changing its
files. Only aliases created by this process may bypass the ordinary no-link
check; input/output paths retain their stricter rules.
"""
import atexit
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
from uuid import uuid4

_aliases = {}
_lock = threading.RLock()
_working_entries = set()


def _private_directory(base):
    import ctypes
    from ctypes import wintypes

    class SecurityAttributes(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('descriptor', ctypes.c_void_p),
                    ('inherit', wintypes.BOOL)]

    advapi = ctypes.WinDLL('advapi32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    convert.restype = wintypes.BOOL
    create = kernel.CreateDirectoryW
    create.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(SecurityAttributes)]
    create.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    descriptor = ctypes.c_void_p()
    # Protected ACL: the creating owner and SYSTEM only, including children.
    if not convert('D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)', 1,
                   ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        attributes = SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor, False)
        path = base / ('ptb-rt-' + uuid4().hex)
        if not create(str(path), ctypes.byref(attributes)):
            raise ctypes.WinError(ctypes.get_last_error())
        return path
    finally:
        kernel.LocalFree(descriptor)


def _private_parent(*, shortest=False):
    from .io.scratch import no_links
    bases = [Path(tempfile.gettempdir()), Path(os.environ['SystemRoot']) / 'Temp']
    if shortest:
        bases.sort(key=lambda base: len(str(base)))
    last_error = None
    for base in bases:
        if not str(base).isascii() or len(str(base)) > 80:
            continue
        try:
            no_links(base)
            return _private_directory(base)
        except (OSError, ValueError) as error:
            last_error = error
            continue
    raise OSError('无法创建运行所需的临时目录。请检查 Windows 临时文件夹的写入权限，'
                  '或将 TEMP/TMP 临时目录改为较短的英文路径后重新启动。') from last_error


def _directory_link(original, name='r'):
    from .io.scratch import no_links
    import _winapi
    no_links(original)
    if not original.is_dir():
        raise FileNotFoundError(str(original))
    if name not in ('r', '_internal'):
        raise ValueError('Invalid private directory entry')
    parent = _private_parent()
    link = parent / name
    created = False
    try:
        _winapi.CreateJunction(str(original), str(link))
        created = True
        if link.resolve() != original.resolve() or not link.samefile(original):
            raise ValueError('Native directory alias target mismatch')
        return link
    except BaseException:
        # Under RedirectionGuard even exists() traverses the junction and fails.
        # RemoveDirectory opens the entry itself and does not follow its target.
        if created:
            os.rmdir(link)
        parent.rmdir()
        raise


def _link_or_copy_file(original, destination):
    import hashlib
    import shutil
    from .io.scratch import no_links
    no_links(original)
    try:
        try:
            os.link(original, destination)
        except OSError:
            shutil.copyfile(original, destination)
            with original.open('rb') as source, destination.open('rb') as copied:
                if hashlib.file_digest(source, 'sha256').digest() != hashlib.file_digest(copied, 'sha256').digest():
                    raise ValueError('Frozen file copy differs')
        else:
            if not destination.samefile(original):
                raise ValueError('Frozen file link differs')
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _remove_directory_mirror(directory, files, directories):
    for path in reversed(files):
        path.unlink(missing_ok=True)
    for path in reversed(directories):
        path.rmdir()
    directory.parent.rmdir()


def _directory_mirror(original):
    from .io.scratch import no_links
    no_links(original)
    if not original.is_dir():
        raise FileNotFoundError(str(original))
    # Keep bundled runtime parents below their short-entry threshold too.
    parent = _private_parent(shortest=True)
    directory = parent / '_internal'
    files = []
    directories = []
    try:
        directory.mkdir()
        directories.append(directory)
        pending = [(original, directory)]
        while pending:
            source, target = pending.pop()
            with os.scandir(source) as entries:
                for item in entries:
                    info = item.stat(follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                        raise ValueError('Reparse paths are not allowed')
                    source_path = Path(item.path)
                    target_path = target / item.name
                    if stat.S_ISDIR(info.st_mode):
                        target_path.mkdir()
                        directories.append(target_path)
                        pending.append((source_path, target_path))
                    elif stat.S_ISREG(info.st_mode):
                        _link_or_copy_file(source_path, target_path)
                        files.append(target_path)
                    else:
                        raise ValueError('Unsupported frozen resource type')
        return directory, files, directories
    except BaseException:
        _remove_directory_mirror(directory, files, directories)
        raise


class DirectoryEntry:
    """Keep a private short directory entry alive until an owned child exits."""
    def __init__(self, directory):
        original = Path(directory).absolute()
        self.path = original
        self.owned = False
        if sys.platform == 'win32' and (not str(original).isascii() or len(str(original)) >= 130):
            self.path = _directory_link(original)
            self.owned = True
            with _lock:
                _working_entries.add(self)

    def close(self):
        if self.owned:
            try:
                os.rmdir(self.path)
                self.path.parent.rmdir()
            except OSError:
                return
            self.owned = False
            with _lock:
                _working_entries.discard(self)


class FrozenApplicationEntry:
    """Keep a frozen executable's canonical path short before Python starts.

    PyInstaller resolves executable junctions before locating python311.dll.
    The small EXE must therefore be a hard link (or verified copy); its data
    directory can remain a junction to the fully validated cache. A process
    whose RedirectionGuard rejects that junction gets a private plain tree.
    """
    def __init__(self, executable):
        original = Path(executable).absolute()
        self.path = original
        self.link = None
        self.mirror = None
        self._mirror_files = []
        self._mirror_directories = []
        if sys.platform != 'win32' or (str(original).isascii() and len(str(original.parent)) < 130):
            return
        from .io.scratch import no_links
        no_links(original)
        try:
            try:
                self.link = _directory_link(original.parent / '_internal', '_internal')
            except OSError as error:
                if getattr(error, 'winerror', None) != 448:
                    raise
                self.mirror, self._mirror_files, self._mirror_directories = _directory_mirror(original.parent / '_internal')
            internal = self.link if self.link is not None else self.mirror
            self.path = internal.parent / original.name
            _link_or_copy_file(original, self.path)
            with _lock:
                _working_entries.add(self)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.link is not None or self.mirror is not None:
            try:
                self.path.unlink(missing_ok=True)
                if self.link is not None:
                    os.rmdir(self.link)
                    self.link.parent.rmdir()
                else:
                    _remove_directory_mirror(self.mirror, self._mirror_files, self._mirror_directories)
            except OSError:
                return
            self.link = None
            self.mirror = None
            self._mirror_files = []
            self._mirror_directories = []
            with _lock:
                _working_entries.discard(self)


def runtime_entry(executable):
    """Return a short ASCII executable path to the exact validated runtime."""
    original = Path(executable).absolute()
    if sys.platform != 'win32' or (str(original).isascii() and len(str(original.parent)) < 130):
        return str(original)
    from .io.scratch import no_links
    no_links(original)
    with _lock:
        for alias, (source, _) in _aliases.items():
            if source == original:
                checked_runtime(alias)
                return str(alias)
        junction = _directory_link(original.parent)
        alias = junction / original.name
        _aliases[alias] = (original, junction.parent)
        return str(alias)


def checked_runtime(executable):
    """Validate ordinary paths or this process's exact registered alias."""
    from .io.scratch import no_links
    path = Path(executable).absolute()
    with _lock:
        record = _aliases.get(path)
        if record is None:
            no_links(path)
            return path
        original, parent = record
        no_links(parent)
        no_links(original)
        if path.parent.resolve() != original.parent.resolve() or not path.samefile(original):
            raise ValueError('Native runtime alias target changed')
        return path


def close_runtime_entries():
    """Remove only this process's junctions and their empty private parents."""
    with _lock:
        for entry in list(_working_entries):
            entry.close()
        for alias, (_, parent) in list(_aliases.items()):
            try:
                # RemoveDirectory removes the junction itself, never its target.
                os.rmdir(alias.parent)
                parent.rmdir()
            except OSError:
                continue
            del _aliases[alias]


atexit.register(close_runtime_entries)
