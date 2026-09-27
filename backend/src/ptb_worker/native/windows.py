"""M01 source_id M01-WIN32: suspended launch, job memory bound, owned handles.

Only launches explicit trusted host argv. No process-name/port based termination.
"""
import ctypes as c
from ctypes import wintypes as w
import os
import subprocess
import time

if os.name != 'nt': raise ImportError('Windows native adapter is unavailable on this platform')
k=c.WinDLL('kernel32',use_last_error=True)


class Startup(c.Structure):
    _fields_=[('cb',w.DWORD),('reserved',w.LPWSTR),('desktop',w.LPWSTR),('title',w.LPWSTR),
        ('x',w.DWORD),('y',w.DWORD),('xs',w.DWORD),('ys',w.DWORD),('xc',w.DWORD),('yc',w.DWORD),
        ('fill',w.DWORD),('flags',w.DWORD),('show',w.WORD),('reserved2size',w.WORD),('reserved2',c.c_void_p),
        ('stdin',w.HANDLE),('stdout',w.HANDLE),('stderr',w.HANDLE)]


class ProcessInfo(c.Structure):
    _fields_=[('process',w.HANDLE),('thread',w.HANDLE),('pid',w.DWORD),('tid',w.DWORD)]


class BasicLimit(c.Structure):
    _fields_=[('per_process_time',c.c_int64),('per_job_time',c.c_int64),('flags',w.DWORD),
        ('min_working',c.c_size_t),('max_working',c.c_size_t),('active_processes',w.DWORD),
        ('affinity',c.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]


class IoCounters(c.Structure):
    _fields_=[(name,c.c_uint64) for name in ['reads','writes','other','read_bytes','write_bytes','other_bytes']]


class ExtendedLimit(c.Structure):
    _fields_=[('basic',BasicLimit),('io',IoCounters),('process_memory',c.c_size_t),('job_memory',c.c_size_t),
         ('peak_process',c.c_size_t),('peak_job',c.c_size_t)]


class JobAccounting(c.Structure):
    _fields_=[('user',c.c_int64),('kernel',c.c_int64),('period_user',c.c_int64),('period_kernel',c.c_int64),
              ('faults',w.DWORD),('total',w.DWORD),('active',w.DWORD),('terminated',w.DWORD)]


def api(name,args,restype=w.BOOL):
    f=getattr(k,name);f.argtypes=args;f.restype=restype;return f


create_job=api('CreateJobObjectW',[c.c_void_p,w.LPCWSTR],w.HANDLE)
set_job=api('SetInformationJobObject',[w.HANDLE,c.c_int,c.c_void_p,w.DWORD])
assign_job=api('AssignProcessToJobObject',[w.HANDLE,w.HANDLE])
terminate_job=api('TerminateJobObject',[w.HANDLE,w.UINT])
close=api('CloseHandle',[w.HANDLE])
wait=api('WaitForSingleObject',[w.HANDLE,w.DWORD],w.DWORD)
resume=api('ResumeThread',[w.HANDLE],w.DWORD)
exit_code=api('GetExitCodeProcess',[w.HANDLE,c.POINTER(w.DWORD)])
terminate_process=api('TerminateProcess',[w.HANDLE,w.UINT])
open_process=api('OpenProcess',[w.DWORD,w.BOOL,w.DWORD],w.HANDLE)
in_job=api('IsProcessInJob',[w.HANDLE,w.HANDLE,c.POINTER(w.BOOL)])
create_process=api('CreateProcessW',[w.LPCWSTR,w.LPWSTR,c.c_void_p,c.c_void_p,w.BOOL,w.DWORD,
    c.c_void_p,w.LPCWSTR,c.POINTER(Startup),c.POINTER(ProcessInfo)])


def checked(value):
    if not value: raise c.WinError(c.get_last_error())
    return value


class OwnedProcess:
    def __init__(self, argv, cwd, memory_bytes):
        self.job=None;self.info=ProcessInfo();self.closed=False
        try:
            self.job=checked(create_job(None,None))
            limits=ExtendedLimit()
            # KILL_ON_JOB_CLOSE | PROCESS_MEMORY | JOB_MEMORY; aggregate includes descendants.
            limits.basic.flags=0x2000|0x100|0x200
            limits.process_memory=memory_bytes;limits.job_memory=memory_bytes
            checked(set_job(self.job,9,c.byref(limits),c.sizeof(limits)))
            startup=Startup();startup.cb=c.sizeof(startup)
            command=c.create_unicode_buffer(subprocess.list2cmdline([str(x) for x in argv]))
            checked(create_process(str(argv[0]),command,None,None,False,0x08000000|4,None,str(cwd),c.byref(startup),c.byref(self.info)))
            checked(assign_job(self.job,self.info.process))
            if resume(self.info.thread)==0xffffffff: raise c.WinError(c.get_last_error())
            close(self.info.thread);self.info.thread=None
        except BaseException:
            self.close();raise

    @property
    def pid(self): return self.info.pid

    def memory_peak(self):
        query=api('QueryInformationJobObject',[w.HANDLE,c.c_int,c.c_void_p,w.DWORD,c.c_void_p])
        limits=ExtendedLimit()
        checked(query(self.job,9,c.byref(limits),c.sizeof(limits),None))
        return int(limits.peak_job)

    def poll(self):
        if wait(self.info.process,0)==258: return None
        value=w.DWORD();checked(exit_code(self.info.process,c.byref(value)))
        return value.value

    def owns_pid(self,pid):
        # Windows venv launchers start the interpreter as a child. Authenticate
        # membership of this exact job (including descendants), not a process name.
        if pid==self.pid:return True  # owning handle prevents PID reuse until close
        handle=open_process(0x1000,False,pid)
        if not handle:return False
        try:
            member=w.BOOL()
            return bool(in_job(handle,self.job,c.byref(member)) and member.value)
        finally:close(handle)

    def close(self):
        if self.closed:return
        self.closed=True
        if self.job: terminate_job(self.job,1)
        if self.info.process:
            # Also covers the suspended process if assignment failed.
            terminate_process(self.info.process,1)
            wait(self.info.process,3000);close(self.info.process)
        if self.info.thread:close(self.info.thread)
        self.group_cleaned=False
        if self.job:
            query=api('QueryInformationJobObject',[w.HANDLE,c.c_int,c.c_void_p,w.DWORD,c.c_void_p])
            until=time.monotonic()+3
            while time.monotonic()<until:
                info=JobAccounting()
                if not query(self.job,1,c.byref(info),c.sizeof(info),None):break
                if info.active==0:self.group_cleaned=True;break
                time.sleep(.01)
            close(self.job)


create_pipe=api('CreateNamedPipeW',[w.LPCWSTR,w.DWORD,w.DWORD,w.DWORD,w.DWORD,w.DWORD,w.DWORD,c.c_void_p],w.HANDLE)
connect_pipe=api('ConnectNamedPipe',[w.HANDLE,c.c_void_p])
read_file=api('ReadFile',[w.HANDLE,c.c_void_p,w.DWORD,c.POINTER(w.DWORD),c.c_void_p])
client_pid=api('GetNamedPipeClientProcessId',[w.HANDLE,c.POINTER(w.ULONG)])


class InputPipe:
    def __init__(self):
        from uuid import uuid4
        self.name='\\\\.\\pipe\\ptb-m01-'+uuid4().hex
        self.peer=None
        # inbound | first instance; byte/nonblocking | reject remote clients.
        self.handle=create_pipe(self.name,1|0x80000,1|8,1,4096,4096,0,None)
        if self.handle==c.c_void_p(-1).value: raise c.WinError(c.get_last_error())

    def read(self,size,owner):
        connect_pipe(self.handle,None)
        peer=w.ULONG()
        identified=client_pid(self.handle,c.byref(peer))
        if identified:
            if self.peer is None:
                if not owner.owns_pid(peer.value):raise ValueError('Unexpected pipe client')
                self.peer=peer.value
            elif self.peer!=peer.value:raise ValueError('Unexpected pipe client')
        buf=c.create_string_buffer(size);n=w.DWORD()
        ok=read_file(self.handle,buf,size,c.byref(n),None)
        error=c.get_last_error()
        if not ok and error not in [109,232,233,536]: raise c.WinError(error)
        if n.value and self.peer is None: raise ValueError('Unverified pipe client')
        return buf.raw[:n.value],error==109

    def close(self):
        if self.handle:
            close(self.handle);self.handle=None
