"""One bounded, owner-scoped transient EGG session per API instance.

No database or result files. Closing the API's Windows Job also kills this
descendant. Idle expiry and RPC deadlines bound lifetime independently of UI.
"""
import atexit
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from .spectrogram_preview import PreviewError


class InteractivePreview:
    def __init__(self, memory_bytes=3_000_000_000, idle_seconds=300, *, reaper_binary=None):
        self.memory_bytes, self.idle_seconds = memory_bytes, idle_seconds
        self.lock = threading.RLock()
        self.process = self.job = self.timer = None
        self.owner = self.token = None
        self.touched = 0
        self.reaper_binary = str(reaper_binary) if reaper_binary else None
        self.native_scratch = None
        self.source_scratch = None
        self.long_source = False
        self.linux_raw = None
        self.linux_audio_key = None
        self.linux_cache_fd = None
        self.linux_source_fd = None
        atexit.register(self.close)

    def _rpc(self, header, raw=b'', timeout=30):
        result = queue.Queue(maxsize=1)
        process = self.process
        def exchange():
            try:
                process.stdin.write(json.dumps(header).encode()+b'\n'+raw)
                process.stdin.flush()
                size = int(process.stdout.readline(32))
                if not 0 < size <= 64_000_000: raise ValueError()
                value = process.stdout.read(size)
                if len(value) != size: raise ValueError()
                result.put(json.loads(value))
            except (OSError, ValueError): result.put({'error':'egg_preview_failed'})
        threading.Thread(target=exchange, daemon=True).start()
        try: value = result.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise PreviewError('preview_timeout', 503) from None
        if 'error' in value: raise PreviewError(value['error'])
        self.touched = time.monotonic()
        return value

    def open(self, owner, raw, *, source_path=None, source_scratch=None):
        if source_path is None and not 0 < len(raw) <= 64_000_000: raise PreviewError('egg_input_budget', 413)
        if sys.platform=='linux':
            if source_path is not None:raise PreviewError('egg_input_budget',413)
            with self.lock:
                if self.linux_raw is not None and self.owner!=owner:raise PreviewError('preview_busy',429)
                self.close()
                value=self._linux_rpc(raw)
                if os.name=='posix':
                    try:
                        import fcntl
                        from .native.anonymous_memory import create, ADD_SEALS, IMMUTABLE, SEAL_SEAL
                        self.linux_cache_fd=create('ptb-egg-state-'+uuid.uuid4().hex)
                        os.fchmod(self.linux_cache_fd,0o600)
                        self.linux_source_fd=create('ptb-egg-source-'+uuid.uuid4().hex,sealable=True)
                        os.fchmod(self.linux_source_fd,0o400)
                        with os.fdopen(os.dup(self.linux_source_fd),'wb') as stream:stream.write(raw)
                        fcntl.fcntl(self.linux_source_fd,ADD_SEALS,IMMUTABLE|SEAL_SEAL)
                    except OSError:
                        for name in ('linux_cache_fd','linux_source_fd'):
                            fd=getattr(self,name)
                            if fd is not None:os.close(fd);setattr(self,name,None)
                self.linux_raw=b'' if self.linux_source_fd is not None else raw;self.owner,self.token=owner,str(uuid.uuid4());self.touched=time.monotonic();self._schedule_expiry()
                return dict(session_id=self.token,**value)
        if os.name != 'nt': raise PreviewError('preview_platform_unverified', 503)
        from .resource_profiles import selected_profile
        from .io.limits import FormatError
        try:
            if selected_profile().shared_admission: raise PreviewError('preview_platform_unverified',503)
        except FormatError: raise PreviewError('preview_platform_unverified',503) from None
        from .native import windows as win
        from .egg_runtime import command
        with self.lock:
            self.close()
            self.source_scratch = source_scratch
            try:
                args = command('', '')[:-2]+['--interactive']
                self.job = win.checked(win.create_job(None, None))
                limits = win.ExtendedLimit()
                limits.basic.flags = 0x2000 | 0x100 | 0x200
                limits.process_memory = limits.job_memory = self.memory_bytes
                win.checked(win.set_job(self.job, 9, win.c.byref(limits), win.c.sizeof(limits)))
                self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW,
                    env={**os.environ, 'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
                ready = queue.Queue()
                process = self.process
                threading.Thread(target=lambda:ready.put(process.stdout.readline(32)), daemon=True).start()
                pid = int(ready.get(timeout=5))
                handle = win.checked(win.open_process(0x0101, False, pid))
                try: win.checked(win.assign_job(self.job, handle))
                finally: win.close(handle)
                header=dict(source_path=str(source_path)) if source_path else dict(size=len(raw))
                if self.reaper_binary:
                    import tempfile
                    from .io.scratch import Scratch
                    from .egg_f0 import NATIVE_BYTES
                    # Local transient session only; fixed bounded owner slot,
                    # closed after the child/process group on every exit path.
                    self.native_scratch=Scratch(tempfile.gettempdir(),NATIVE_BYTES)
                    header.update(native_scratch=str(self.native_scratch.create(b'','.wav')),reaper_binary=self.reaper_binary)
                value = self._rpc(header, raw, timeout=120 if source_path else 30)
                self.long_source = bool(value.get('overview_base64'))
                self.owner, self.token = owner, str(uuid.uuid4())
                self._schedule_expiry()
                return dict(session_id=self.token, **value)
            except Exception as exc:
                self.close()
                if isinstance(exc, PreviewError): raise
                raise PreviewError(getattr(exc,'code','egg_runtime_unavailable'), 503) from None

    def update(self, owner, token, config, *, auto_db=False,need_audio=False):
        with self.lock:
            if sys.platform=='linux':
                if self.owner!=owner or self.token!=str(token) or self.linux_raw is None:raise PreviewError('egg_preview_expired',410)
                options={}
                if auto_db:options['auto_db']=True
                if need_audio:options['need_audio']=True
                result=self._linux_rpc(self.linux_raw,config,**options)
                if result.get('audio_base64'):self.linux_audio_key=bool(config.get('flip_channels',False))
                self.touched=time.monotonic();return result
            if self.owner != owner or self.token != str(token) or self.process is None:
                raise PreviewError('egg_preview_expired', 410)
            try: return self._rpc(dict(config=config),timeout=120 if self.long_source else 30)
            except PreviewError:
                # Keep a healthy child after a validation/scientific error.
                if self.process is not None and self.process.poll() is not None: self.close()
                raise

    def release(self, owner, token):
        with self.lock:
            if self.owner == owner and self.token == str(token): self.close()

    def _schedule_expiry(self):
        self.timer = threading.Timer(self.idle_seconds, self._expire)
        self.timer.daemon = True
        self.timer.start()

    def _expire(self):
        with self.lock:
            if self.process is None and self.linux_raw is None: return
            if time.monotonic()-self.touched >= self.idle_seconds: self.close()
            else: self._schedule_expiry()

    def close(self):
        with self.lock:
            self.linux_raw=None
            self.linux_audio_key=None
            if self.linux_cache_fd is not None:
                os.close(self.linux_cache_fd);self.linux_cache_fd=None
            if self.linux_source_fd is not None:
                os.close(self.linux_source_fd);self.linux_source_fd=None
            if self.timer: self.timer.cancel(); self.timer = None
            if self.job:
                from .native import windows as win
                win.terminate_job(self.job, 1); win.close(self.job); self.job = None
            if self.process:
                process, self.process = self.process, None
                if process.poll() is None: process.terminate()
                process.wait(timeout=5)
                process.stdin.close(); process.stdout.close()
            self.owner = self.token = None
            if self.native_scratch:
                scratch,self.native_scratch=self.native_scratch,None
                scratch.close()
            if self.source_scratch:
                scratch,self.source_scratch=self.source_scratch,None
                scratch.cleanup()
            self.long_source = False

    def _linux_rpc(self,raw,config=None,*,auto_db=False,need_audio=False):
        from pathlib import Path
        from .native.linux_runtime import command,load_profile
        from .native.posix import run_bounded
        from .io.limits import Limits,LimitError,FormatError
        scratch=None
        try:
            _,profile=load_profile()
            header=dict(size=len(raw),config=config)
            if config is not None:
                if self.linux_source_fd is not None:
                    header.update(size=os.fstat(self.linux_source_fd).st_size,source_memory=f'/proc/{os.getpid()}/fd/{self.linux_source_fd}')
                header.update(audio_ready=self.linux_audio_key==bool(config.get('flip_channels',False)) if self.linux_audio_key is not None else False,auto_db=auto_db)
                if need_audio:header['audio_ready']=False
                if self.linux_cache_fd is not None:header['checkpoint']=f'/proc/{os.getpid()}/fd/{self.linux_cache_fd}'
            if config and config.get('keep_reaper_f0'):
                from .io.scratch import Scratch
                from .egg_f0 import NATIVE_BYTES
                binary=profile.get('reaper_binary')
                if not binary:raise PreviewError('egg_reaper_unavailable',503)
                scratch=Scratch(profile['cache'],NATIVE_BYTES)
                header.update(native_scratch=str(scratch.create(b'','.wav')),reaper_binary=binary)
            encoded=json.dumps(header).encode()+b'\n'
            output=run_bounded(command('egg_interactive','--preview'),encoded+(b'' if 'source_memory' in header else raw),Path(profile['cache']),
                Limits(input_bytes=64_032_768,output_bytes=64_000_000,process_bytes=min(self.memory_bytes,1_000_000_000),timeout_seconds=60))
            result=json.loads(output)
            if 'error' in result:raise PreviewError(result['error'])
            return result
        except PreviewError:raise
        except LimitError as exc:raise PreviewError('preview_timeout' if str(exc)=='native_timeout' else 'egg_preview_failed',503) from None
        except (OSError,ValueError,FormatError):raise PreviewError('egg_preview_failed',503) from None
        finally:
            if scratch:scratch.close()
