"""Read-only, memory-only display child, gated before importing native libraries.

Windows parent owns a KILL_ON_JOB_CLOSE Job. No output files, shell commands,
user-supplied paths, global process termination, or persistent science tasks.
"""
import base64
from ptb_worker.process_entry import command
from contextlib import contextmanager
import json
import os
import queue
import subprocess
import sys
import threading

MAX_BYTES=64_000_000
_slot=threading.BoundedSemaphore(1)


class PreviewError(ValueError):
    def __init__(self,code,status=422):self.code,self.status=code,status;super().__init__(code)


@contextmanager
def preview_slot():
    if not _slot.acquire(False):raise PreviewError('preview_busy',429)
    try:yield
    finally:_slot.release()


def render(payload,*,channel,start,end,width=800,timeout=20):
    if len(payload)>MAX_BYTES:raise PreviewError('preview_too_large',413)
    if os.name!='nt':raise PreviewError('preview_platform_unverified',503)
    from .native import windows as win
    process=None;job=None
    try:
        job=win.checked(win.create_job(None,None));limits=win.ExtendedLimit()
        limits.basic.flags=0x2000|0x100|0x200
        limits.process_memory=1_000_000_000;limits.job_memory=1_000_000_000
        win.checked(win.set_job(job,9,win.c.byref(limits),win.c.sizeof(limits)))
        process=subprocess.Popen(command('ptb_worker.spectrogram_preview'),stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW,
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        ready=queue.Queue()
        threading.Thread(target=lambda:ready.put(process.stdout.readline(64)),daemon=True).start()
        pid=int(ready.get(timeout=5))
        # The trusted child waits for its header before imports or reading audio.
        # A Windows venv launcher can have a distinct interpreter PID.
        handle=win.checked(win.open_process(0x0101,False,pid))
        try:win.checked(win.assign_job(job,handle))
        finally:win.close(handle)
        header=json.dumps(dict(size=len(payload),channel=channel,start=start,end=end,width=width)).encode()+b'\n'
        try:output,_=process.communicate(header+payload,timeout=timeout)
        except subprocess.TimeoutExpired:raise PreviewError('preview_timeout',503) from None
        if process.returncode!=0 or len(output)>512_000:raise PreviewError('preview_resource_failed',503)
        result=json.loads(output)
        if 'error' in result:raise PreviewError(result['error'],503 if result['error'] in ('preview_runtime_unavailable','preview_memory_exceeded') else 422)
        return result
    except PreviewError:raise
    except (OSError,ValueError,queue.Empty):raise PreviewError('preview_resource_failed',503) from None
    finally:
        if job:win.terminate_job(job,1);win.close(job)
        if process:
            if process.poll() is None:process.terminate()
            process.wait(timeout=5)
            process.stdin.close();process.stdout.close()


def child():
    print(os.getpid(),flush=True)
    try:
        header=json.loads(sys.stdin.buffer.readline(8192));size=header.pop('size')
        if type(size)!=int or not 0<size<=MAX_BYTES:raise ValueError()
        payload=sys.stdin.buffer.read(size)
        if len(payload)!=size:raise ValueError()
        from .io.audio import decode_wav
        from .io.limits import Limits
        from phonetic_core.spectrogram import spectrogram_preview
        audio=decode_wav(payload,Limits(input_bytes=MAX_BYTES,samples=32_000_000,channels=32))
        result=spectrogram_preview(audio,**header)
        result.pop('_power');result.pop('_frequencies')
        result['pixels_base64']=base64.b64encode(result.pop('pixels')).decode('ascii')
    except ImportError:result={'error':'preview_runtime_unavailable'}
    except MemoryError:result={'error':'preview_memory_exceeded'}
    except Exception:result={'error':'invalid_spectrogram_input'}
    output=json.dumps(result,allow_nan=False).encode()
    if len(output)>512_000:raise SystemExit(2)
    sys.stdout.buffer.write(output);sys.stdout.buffer.flush()


if __name__=='__main__':child()
