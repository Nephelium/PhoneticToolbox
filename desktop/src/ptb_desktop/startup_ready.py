"""A per-launch Win32 event, signalled only by the owning workbench window."""
import ctypes
from ctypes import wintypes
import os
import re
import uuid

ENVIRONMENT = 'PTB_STARTUP_READY_EVENT'
NAME = re.compile(r'Local\\PTBStartupReady-[0-9a-f]{32}')


def kernel():
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateEventW.argtypes=(ctypes.c_void_p,wintypes.BOOL,wintypes.BOOL,wintypes.LPCWSTR)
    api.CreateEventW.restype=wintypes.HANDLE
    api.OpenEventW.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.LPCWSTR)
    api.OpenEventW.restype=wintypes.HANDLE
    api.SetEvent.argtypes=(wintypes.HANDLE,);api.SetEvent.restype=wintypes.BOOL
    api.WaitForSingleObject.argtypes=(wintypes.HANDLE,wintypes.DWORD)
    api.WaitForSingleObject.restype=wintypes.DWORD
    api.CloseHandle.argtypes=(wintypes.HANDLE,);api.CloseHandle.restype=wintypes.BOOL
    return api


class ReadyEvent:
    def __init__(self):
        self.name='Local\\PTBStartupReady-'+uuid.uuid4().hex
        self.api=kernel();self.handle=self.api.CreateEventW(None,True,False,self.name)
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error()==183:
            self.close();raise RuntimeError('Startup event already exists')

    def is_set(self):
        result=self.api.WaitForSingleObject(self.handle,0)
        if result==0:return True
        if result==258:return False
        raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:self.api.CloseHandle(self.handle);self.handle=None


def signal_ready():
    name=os.environ.get(ENVIRONMENT,'')
    if os.name!='nt' or not NAME.fullmatch(name):return False
    api=kernel();handle=api.OpenEventW(2,False,name)
    if not handle:return False  # The launcher may already have exited.
    try:
        if not api.SetEvent(handle):raise ctypes.WinError(ctypes.get_last_error())
        return True
    finally:api.CloseHandle(handle)
