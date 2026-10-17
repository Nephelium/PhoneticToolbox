"""Linux memfd capabilities, including Python builds without os.memfd_create."""
import ctypes
import os

# Linux UAPI values (include/uapi/linux/{memfd,fcntl}.h).
CLOEXEC = 0x0001
ALLOW_SEALING = 0x0002
ADD_SEALS = 1033
GET_SEALS = 1034
SEAL_SEAL = 0x0001
SEAL_SHRINK = 0x0002
SEAL_GROW = 0x0004
SEAL_WRITE = 0x0008
IMMUTABLE = SEAL_WRITE | SEAL_GROW | SEAL_SHRINK


def create(name, *, sealable=False):
    flags = CLOEXEC | (ALLOW_SEALING if sealable else 0)
    if hasattr(os, 'memfd_create'):
        return os.memfd_create(name, flags)
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, 'memfd_create', None)
    if function is None:
        raise OSError('memfd_create unavailable')
    function.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    fd = function(name.encode('ascii'), flags)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd
