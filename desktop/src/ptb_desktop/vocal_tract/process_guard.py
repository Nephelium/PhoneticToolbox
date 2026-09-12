"""Windows kernel ownership: closing this job terminates only its assigned child."""
import ctypes as C
from ctypes import wintypes as W
import os


def kernel():
    k=C.WinDLL('kernel32',use_last_error=True)
    k.CreateJobObjectW.argtypes=[C.c_void_p,W.LPCWSTR];k.CreateJobObjectW.restype=W.HANDLE
    k.SetInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD];k.SetInformationJobObject.restype=W.BOOL
    k.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE];k.AssignProcessToJobObject.restype=W.BOOL
    k.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];k.OpenProcess.restype=W.HANDLE
    k.WaitForSingleObject.argtypes=[W.HANDLE,W.DWORD];k.WaitForSingleObject.restype=W.DWORD
    k.CloseHandle.argtypes=[W.HANDLE];k.CloseHandle.restype=W.BOOL
    return k


class OwnedJob:
    def __init__(self, process):
        self.handle=None
        if os.name!='nt':return
        class Basic(C.Structure):
            _fields_=[('per_process',C.c_int64),('per_job',C.c_int64),('flags',W.DWORD),('min_ws',C.c_size_t),('max_ws',C.c_size_t),('active',W.DWORD),('affinity',C.c_size_t),('priority',W.DWORD),('scheduling',W.DWORD)]
        class Extended(C.Structure):
            _fields_=[('basic',Basic),('io',C.c_uint64*6),('process_memory',C.c_size_t),('job_memory',C.c_size_t),('peak_process',C.c_size_t),('peak_job',C.c_size_t)]
        self.k=kernel();handle=self.k.CreateJobObjectW(None,None)
        if not handle:raise C.WinError(C.get_last_error())
        info=Extended();info.basic.flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.k.SetInformationJobObject(handle,9,C.byref(info),C.sizeof(info)) or not self.k.AssignProcessToJobObject(handle,int(process._handle)):
            error=C.get_last_error();self.k.CloseHandle(handle);raise C.WinError(error)
        self.handle=handle

    def close(self):
        if self.handle:self.k.CloseHandle(self.handle);self.handle=None


def watch_parent(pid):
    """Open a real process handle once: PID reuse cannot keep an orphan alive."""
    import threading
    import time
    if os.name=='nt':
        k=kernel();handle=k.OpenProcess(0x100000,False,pid)
        if not handle:raise RuntimeError('主应用已经退出')
        def wait():
            try:
                if k.WaitForSingleObject(handle,0xFFFFFFFF)==0:os._exit(0)
            finally:k.CloseHandle(handle)
    else:
        def wait():
            while os.getppid()==pid:time.sleep(.5)
            os._exit(0)
    threading.Thread(target=wait,name='vocal-parent-watch',daemon=True).start()
