"""Directory-scoped output I/O: Windows deny-delete handle, POSIX dir_fd.

POSIX rename replaces existing files, so no-clobber publication uses an atomic
hard link followed by removal of our temporary name. No shell or global cwd.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import stat
from .file_provider import checked_path, FileAccessError, identity


class PinnedDirectory:
    def __init__(self, root, descriptor=None):
        self.root, self.descriptor = root, descriptor
        self.identity = identity(os.fstat(descriptor) if descriptor is not None else root.stat())

    def _name(self, path):
        path=Path(path)
        if path.is_absolute():
            if path.parent!=self.root:raise FileAccessError('输出超出授权目录。')
            name=path.name
        else:name=str(path)
        if name in ('','.','..') or any(c in name for c in '/\\:'):
            raise FileAccessError('输出文件名无效。')
        return name

    def check(self):
        if checked_path(self.root)!=self.root or identity(self.root.stat())!=self.identity:
            raise FileAccessError('输出目录已变化，请重新选择。')

    def stat(self, path):
        name=self._name(path)
        if self.descriptor is None:return (self.root/name).lstat()
        return os.stat(name,dir_fd=self.descriptor,follow_symlinks=False)

    def exists(self, path):
        try:self.stat(path);return True
        except FileNotFoundError:return False

    def open(self, path, mode):
        if mode not in ('rb','xb'):raise ValueError('Only read or exclusive creation is permitted')
        name=self._name(path);self.check()
        flags=os.O_RDONLY if mode=='rb' else os.O_WRONLY|os.O_CREAT|os.O_EXCL
        flags|=getattr(os,'O_BINARY',0)|getattr(os,'O_NOFOLLOW',0)
        fd=os.open(self.root/name,flags,0o600) if self.descriptor is None else os.open(name,flags,0o600,dir_fd=self.descriptor)
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or getattr(info,'st_file_attributes',0)&0x400:
                raise FileAccessError('输出目标不是独立普通文件。')
            return os.fdopen(fd,mode)
        except BaseException:
            os.close(fd);raise

    def unlink(self, path, expected=None):
        # Cleanup stays bound to the original directory even if it was renamed.
        name=self._name(path)
        try:info=self.stat(name)
        except FileNotFoundError:return
        if expected is not None and identity(info)!=expected:return
        if self.descriptor is None:(self.root/name).unlink()
        else:os.unlink(name,dir_fd=self.descriptor)

    def publish(self, temporary, target, *, replace=False):
        source,dest=self._name(temporary),self._name(target);self.check()
        if self.descriptor is None:
            (os.replace if replace else os.rename)(self.root/source,self.root/dest)
        elif replace:
            os.replace(source,dest,src_dir_fd=self.descriptor,dst_dir_fd=self.descriptor)
        else:
            os.link(source,dest,src_dir_fd=self.descriptor,dst_dir_fd=self.descriptor,follow_symlinks=False)
            os.unlink(source,dir_fd=self.descriptor)
        if self.descriptor is not None:os.fsync(self.descriptor)


@contextmanager
def pin_directory(path):
    root=checked_path(path)
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        create=kernel.CreateFileW
        create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        create.restype=wintypes.HANDLE
        handle=create(str(root),0x80000000,3,None,3,0x02200000,None)
        if handle==ctypes.c_void_p(-1).value:raise FileAccessError('无法锁定结果目录，请重新选择。')
        close=kernel.CloseHandle;close.argtypes=[wintypes.HANDLE]
        try:yield PinnedDirectory(root)
        finally:close(handle)
    elif os.name=='posix':
        flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW
        fd=os.open(root.anchor,flags)
        try:
            # Walk from the filesystem root with no-follow at every component.
            for part in root.parts[1:]:
                child=os.open(part,flags,dir_fd=fd)
                os.close(fd);fd=child
            directory=PinnedDirectory(root,fd);directory.check()
            yield directory
        finally:os.close(fd)
    else:
        raise FileAccessError('当前系统尚未提供安全的目录保存适配。')
