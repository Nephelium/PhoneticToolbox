"""Authenticated hosts own these bounded subprocess sessions, never client paths."""
import atexit
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import tempfile
from uuid import uuid4

OPS=frozenset(('meta','status','heartbeat','pose','live','preview','deactivate','audio/settings','audio/test',
    'audio/pull','audio/monitor','keyframes/load','keyframes','presets/load','presets/save','favorites/load','favorites/save',
    'document/validate','document/create','animation/prepare','animation/stop','animation/play','animation/frame','animation/audio','animation/picture',
    'video/begin','video/chunks','video/finish','video/cancel'))


class VocalError(Exception):
    def __init__(self,code,status=422):self.code=code;self.status=status;super().__init__(code)


class Session:
    def __init__(self,resources,identity):
        from .process_entry import command
        self.id=str(uuid4());self.identity=identity;self.touched=time.monotonic();self.pending={};self.guard=threading.RLock();self.write_lock=threading.Lock();self.closed=False;self.video=None;self.generation=0
        self.windows_job=None
        self.scratch=tempfile.TemporaryDirectory(prefix='ptb-vocal-session-')
        self.process=subprocess.Popen(command('ptb_worker.vocal_child','--resources',str(resources)),
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PTB_VOCAL_SCRATCH':self.scratch.name})
        if os.name=='nt':
            from .native import windows as win
            self.windows_job=win.checked(win.create_job(None,None));limits=win.ExtendedLimit()
            limits.basic.flags=0x2000|0x100|0x200;limits.process_memory=limits.job_memory=1_073_741_824
            win.checked(win.set_job(self.windows_job,9,win.c.byref(limits),win.c.sizeof(limits)))
            handle=win.checked(win.open_process(0x0101,False,self.process.pid))
            try:win.checked(win.assign_job(self.windows_job,handle))
            finally:win.close(handle)
        threading.Thread(target=self._read,daemon=True).start()

    def _read(self):
        try:
            while True:
                raw=self.process.stdout.readline(132_000_001)
                if not raw or len(raw)>132_000_000:break
                value=json.loads(raw)
                with self.guard:item=self.pending.pop(value.pop('request_id'),None)
                if item:item.put(value)
        except (OSError,ValueError,KeyError):pass
        finally:
            with self.guard:
                for item in self.pending.values():item.put({'ok':False,'unavailable':True})
                self.pending.clear()

    def invoke(self,op,body,timeout=150,*,generation=None):
        if op not in OPS:raise VocalError('vocal_operation_rejected')
        rid=uuid4().hex;item=queue.Queue(maxsize=1)
        with self.guard:
            if self.closed or self.process.poll() is not None:raise VocalError('vocal_session_expired',410)
            if len(self.pending)>=4:raise VocalError('vocal_busy',429)
            self.pending[rid]=item;self.touched=time.monotonic()
        try:
            raw=json.dumps(dict(request_id=rid,op=op,body=body,generation=generation),allow_nan=False,ensure_ascii=False).encode('utf-8')+b'\n'
            if len(raw)>9_000_000:raise VocalError('vocal_request_too_large',413)
            with self.write_lock:self.process.stdin.write(raw);self.process.stdin.flush()
            try:result=item.get(timeout=timeout)
            except queue.Empty:
                self.close();raise VocalError('vocal_timeout',503) from None
            if result.get('unavailable'):self.close();raise VocalError('vocal_runtime_unavailable',503)
            if not result['ok']:raise VocalError(result['error'])
            return result['value']
        except (OSError,ValueError):self.close();raise VocalError('vocal_runtime_unavailable',503) from None
        finally:
            with self.guard:self.pending.pop(rid,None)

    def close(self):
        with self.guard:
            if self.closed:return
            self.closed=True
            if self.video:self.video.cancel();self.video=None
            if self.process.poll() is None:self.process.kill()
            self.process.wait(timeout=5)
            self.process.stdin.close();self.process.stdout.close()
            if self.windows_job:
                from .native import windows as win
                win.close(self.windows_job);self.windows_job=None
            self.scratch.cleanup()


class VocalSessions:
    def __init__(self,resources,*,max_sessions=2,idle_seconds=1800,valid=lambda identity:True):
        self.resources=Path(resources).resolve();self.max_sessions=max_sessions;self.idle_seconds=idle_seconds;self.valid=valid
        self.sessions={};self.lock=threading.RLock();self.compute=threading.BoundedSemaphore(1);self.stop=threading.Event()
        threading.Thread(target=self._maintain,daemon=True).start();atexit.register(self.close)

    def open(self,identity):
        with self.lock:
            self.sweep()
            # One native state per browser account/project; no unrelated user's state is evicted.
            if sum(s.identity==identity for s in self.sessions.values())>=2 or len(self.sessions)>=self.max_sessions:raise VocalError('vocal_capacity_full',429)
            if not self.valid(identity):raise VocalError('vocal_session_expired',410)
            session=Session(self.resources,identity);self.sessions[session.id]=session
            return {'id':session.id,'idle_seconds':self.idle_seconds}

    def get(self,sid,identity):
        with self.lock:
            s=self.sessions.get(sid)
            if s is None or s.identity!=identity:raise VocalError('vocal_session_not_found',404)
            if s.closed or time.monotonic()-s.touched>=self.idle_seconds or not self.valid(identity):
                self.sessions.pop(sid);s.close();raise VocalError('vocal_session_expired',410)
            s.touched=time.monotonic();return s

    def invoke(self,sid,identity,op,body):
        s=self.get(sid,identity)
        generation=s.generation
        if op in ('heartbeat','status'):
            value=s.invoke(op,body,10);self.get(sid,identity);return value
        if op in ('animation/stop','deactivate') or op=='live' and not body.get('active'):
            with s.guard:s.generation+=1
            return s.invoke(op,body,10)
        if op=='video/cancel':return s.invoke(op,body,10)
        if not self.compute.acquire(timeout=5):raise VocalError('vocal_busy',429)
        try:
            if op in ('preview','animation/prepare') and generation!=s.generation:raise VocalError('vocal_cancelled',409)
            value=s.invoke(op,body,generation=generation)
            self.get(sid,identity) # Fence logout/expiry during a native calculation.
            return value
        finally:self.compute.release()

    def release(self,sid,identity):
        s=self.get(sid,identity)
        with self.lock:self.sessions.pop(sid,None)
        s.close();return {'closed':True}

    def sweep(self):
        with self.lock:
            for sid,s in list(self.sessions.items()):
                if s.closed or time.monotonic()-s.touched>=self.idle_seconds or not self.valid(s.identity):
                    self.sessions.pop(sid);s.close()

    def _maintain(self):
        while not self.stop.wait(5):
            try:self.sweep()
            except Exception:pass # Authentication/DB outage denies new requests; next sweep retries cleanup.

    def close(self):
        self.stop.set()
        with self.lock:
            for s in self.sessions.values():s.close()
            self.sessions.clear()
