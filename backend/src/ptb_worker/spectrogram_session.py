"""One local API's bounded, transient Praat worker; no files or DB state."""
import atexit
import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from .process_entry import command
from .spectrogram_preview import MAX_BYTES, PreviewError, render


class SpectrogramSession:
    def __init__(self, idle_seconds=60):
        self.lock=threading.RLock()
        self.process=self.job=self.timer=None
        self.digest=None
        self.touched=0
        self.idle_seconds=idle_seconds
        atexit.register(self.close)

    def _start(self):
        from .native import windows as win
        self.job=win.checked(win.create_job(None,None))
        limits=win.ExtendedLimit()
        limits.basic.flags=0x2000|0x100|0x200
        limits.process_memory=limits.job_memory=1_000_000_000
        win.checked(win.set_job(self.job,9,win.c.byref(limits),win.c.sizeof(limits)))
        self.process=subprocess.Popen(command('ptb_worker.spectrogram_preview','--interactive'),
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        ready=queue.Queue(maxsize=1)
        process=self.process
        threading.Thread(target=lambda:ready.put(process.stdout.readline(32)),daemon=True).start()
        pid=int(ready.get(timeout=5))
        handle=win.checked(win.open_process(0x0101,False,pid))
        try:win.checked(win.assign_job(self.job,handle))
        finally:win.close(handle)
        self.touched=time.monotonic()
        self._schedule()

    def render(self,payload,*,channel,start,end,width=800,timeout=20):
        if not 0<len(payload)<=MAX_BYTES:raise PreviewError('preview_too_large',413)
        if os.name!='nt':return render(payload,channel=channel,start=start,end=end,width=width,timeout=timeout)
        with self.lock:
            try:
                if self.process is None:self._start()
                digest=hashlib.sha256(payload).digest()
                raw=payload if digest!=self.digest else b''
                header=dict(size=len(raw),channel=channel,start=start,end=end,width=width)
                process=self.process
                ready=queue.Queue(maxsize=1)
                def exchange():
                    try:
                        process.stdin.write(json.dumps(header).encode()+b'\n')
                        if raw:process.stdin.write(raw)
                        process.stdin.flush()
                        size=int(process.stdout.readline(32))
                        if not 0<size<=512_000:raise ValueError('output budget')
                        value=process.stdout.read(size)
                        if len(value)!=size:raise ValueError('short response')
                        ready.put(json.loads(value))
                    except (OSError,ValueError):ready.put({'error':'preview_resource_failed'})
                threading.Thread(target=exchange,daemon=True).start()
                try:value=ready.get(timeout=timeout)
                except queue.Empty:raise PreviewError('preview_timeout',503) from None
                if 'error' in value:raise PreviewError(value['error'])
                self.digest=digest
                self.touched=time.monotonic()
                return value
            except Exception as exc:
                self.close()
                if isinstance(exc,PreviewError):raise
                raise PreviewError('preview_resource_failed',503) from None

    def _schedule(self,delay=None):
        self.timer=threading.Timer(self.idle_seconds if delay is None else delay,self._expire)
        self.timer.daemon=True
        self.timer.start()

    def _expire(self):
        with self.lock:
            if self.process is None:return
            remaining=self.idle_seconds-(time.monotonic()-self.touched)
            if remaining<=0:self.close()
            else:self._schedule(remaining)

    def close(self):
        with self.lock:
            if self.timer:self.timer.cancel();self.timer=None
            if self.job:
                from .native import windows as win
                win.terminate_job(self.job,1);win.close(self.job);self.job=None
            if self.process:
                process,self.process=self.process,None
                if process.poll() is None:process.terminate()
                process.wait(timeout=5)
                process.stdin.close();process.stdout.close()
            self.digest=None
