"""Instance OS/PG gate and stable scoped locks. Lock order is always outer first."""
from contextlib import contextmanager
import hashlib
import os
import time
from .quota import StorageError


def scoped_key(kind,value):
    raw=hashlib.sha256(('ptb-storage-v1:'+kind+':'+str(value)).encode()).digest()[:8]
    # Separate from existing positive application advisory lock IDs.
    return -1-int.from_bytes(raw,'big')%((1<<63)-1)


@contextmanager
def advisory(conn,key,*,shared=False):
    suffix='_shared' if shared else ''
    conn.execute('SELECT pg_advisory_lock'+suffix+'(%s)',(key,))
    try:yield
    finally:conn.execute('SELECT pg_advisory_unlock'+suffix+'(%s)',(key,))


@contextmanager
def instance_file_lock(path,*,shared):
    with path.open('r+b',buffering=0) as handle:
        if os.name=='nt':
            import ctypes as c
            from ctypes import wintypes as w
            import msvcrt
            class Overlapped(c.Structure):
                _fields_=[('internal',c.c_size_t),('internal_high',c.c_size_t),
                          ('offset',w.DWORD),('offset_high',w.DWORD),('event',w.HANDLE)]
            api=c.WinDLL('kernel32',use_last_error=True)
            api.LockFileEx.argtypes=[w.HANDLE,w.DWORD,w.DWORD,w.DWORD,w.DWORD,c.POINTER(Overlapped)]
            api.LockFileEx.restype=w.BOOL
            api.UnlockFileEx.argtypes=[w.HANDLE,w.DWORD,w.DWORD,w.DWORD,c.POINTER(Overlapped)]
            api.UnlockFileEx.restype=w.BOOL
            overlap=Overlapped();native=msvcrt.get_osfhandle(handle.fileno())
            def acquire():
                if api.LockFileEx(native,1|(0 if shared else 2),0,1,0,c.byref(overlap)):return True
                if c.get_last_error()!=33:raise c.WinError(c.get_last_error())
                return False
            def release():
                if not api.UnlockFileEx(native,0,1,0,c.byref(overlap)):raise c.WinError(c.get_last_error())
        else:
            import fcntl
            def acquire():
                try:fcntl.flock(handle.fileno(),(fcntl.LOCK_SH if shared else fcntl.LOCK_EX)|fcntl.LOCK_NB);return True
                except BlockingIOError:return False
            def release():fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        deadline=time.monotonic()+5
        while not acquire():
            if time.monotonic()>=deadline:raise StorageError('storage_busy',503)
            time.sleep(.01)
        try:yield
        finally:release()
