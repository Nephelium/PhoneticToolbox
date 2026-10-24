"""Persistent macOS owner for the existing EGG session and file-backed reader."""
import os,queue,signal,subprocess,threading,time,uuid
from .darwin import command,_memory,_close_group,load_profile
from ..egg_interactive import InteractivePreview as Base
from ..spectrogram_preview import PreviewError

class InteractivePreview(Base):
    def __init__(self,*args,**kwargs):
        self._monitor_stop=None;self._monitor=None
        super().__init__(*args,**kwargs)
    def open(self,owner,raw,*,source_path=None,source_scratch=None):
        if source_path is None and not 0<len(raw)<=64_000_000:raise PreviewError('egg_input_budget',413)
        with self.lock:
            self.close();self.source_scratch=source_scratch
            try:
                env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PTB_MACOS_CHILD_CPU='1803')
                self.process=subprocess.Popen(command('egg_persistent','--interactive'),stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,env=env,start_new_session=True)
                process=self.process;ready=queue.Queue();threading.Thread(target=lambda:ready.put(process.stdout.readline(32)),daemon=True).start()
                if int(ready.get(timeout=25))!=process.pid:raise ValueError('PID mismatch')
                self._monitor_stop=threading.Event();stop=self._monitor_stop
                def monitor():
                    while not stop.wait(.1) and process.poll() is None:
                        if _memory(process)>self.memory_bytes:
                            try:os.killpg(process.pid,signal.SIGKILL)
                            except ProcessLookupError:pass
                            return
                self._monitor=threading.Thread(target=monitor,daemon=True);self._monitor.start()
                header=dict(source_path=str(source_path)) if source_path else dict(size=len(raw))
                if self.reaper_binary:
                    from ..io.scratch import Scratch
                    from ..egg_f0 import NATIVE_BYTES
                    _,profile=load_profile();self.native_scratch=Scratch(profile['cache'],NATIVE_BYTES)
                    header.update(native_scratch=str(self.native_scratch.create(b'','.wav')),reaper_binary=self.reaper_binary)
                result=self._rpc(header,raw,timeout=120 if source_path else 30)
                self.long_source=bool(result.get('overview_base64'));self.owner,self.token=owner,str(uuid.uuid4());self._schedule_expiry()
                return dict(session_id=self.token,**result)
            except BaseException:
                self.close();raise
    def update(self,owner,token,config,*,auto_db=False,need_audio=False):
        with self.lock:
            if self.owner!=owner or self.token!=str(token) or self.process is None:raise PreviewError('egg_preview_expired',410)
            try:return self._rpc(dict(config=config,auto_db=auto_db,need_audio=need_audio),timeout=120 if self.long_source else 30)
            except PreviewError:
                if self.process is not None and self.process.poll() is not None:self.close()
                raise
    def close(self):
        with self.lock:
            if self.timer:self.timer.cancel();self.timer=None
            if self._monitor_stop:self._monitor_stop.set();self._monitor_stop=None
            if self._monitor:self._monitor.join(timeout=2);self._monitor=None
            if self.process:
                process,self.process=self.process,None;_close_group(process)
            self.owner=self.token=None;self.long_source=False
            if self.native_scratch:self.native_scratch.close();self.native_scratch=None
            if self.source_scratch:self.source_scratch.cleanup();self.source_scratch=None
