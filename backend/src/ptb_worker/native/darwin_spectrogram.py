"""One idle-expiring owned Mac process for the shared Praat preview protocol."""
import atexit,json,os,queue,signal,subprocess,threading,time
from .darwin import command,_memory,_close_group,load_profile
from .darwin_preview_cache import ResultCache
from ..spectrogram_preview import MAX_BYTES,PreviewError


class SpectrogramSession:
    def __init__(self,idle_seconds=60):
        self.lock=threading.RLock();self.process=None;self.timer=None;self.monitor=None
        self.stop=None;self.digest=None;self.touched=0;self.idle_seconds=idle_seconds
        self.cache=ResultCache(max_bytes=2_000_000,max_entries=4,ttl=idle_seconds)
        atexit.register(self.close)

    def _start(self):
        _,profile=load_profile()
        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MPLBACKEND':'Agg','MPLCONFIGDIR':profile['cache']}
        # Lifetime is bounded by idle expiry and aggregate RSS; each RPC has its own wall deadline.
        env.pop('PTB_MACOS_CHILD_CPU',None)
        self.process=subprocess.Popen(command('spectrogram','--interactive'),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=env,start_new_session=True)
        process=self.process;ready=queue.Queue(maxsize=1)
        threading.Thread(target=lambda:ready.put(process.stdout.readline(32)),daemon=True).start()
        if int(ready.get(timeout=25))!=process.pid:raise ValueError('Preview PID mismatch')
        self.stop=threading.Event();stop=self.stop
        def monitor():
            while not stop.wait(.1) and process.poll() is None:
                if _memory(process)>1_000_000_000:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    return
        self.monitor=threading.Thread(target=monitor,daemon=True);self.monitor.start()
        self.touched=time.monotonic();self._schedule(self.idle_seconds)

    def render(self,payload,*,channel,start,end,width=800,timeout=20):
        import hashlib
        if not 0<len(payload)<=MAX_BYTES:raise PreviewError('preview_too_large',413)
        digest=hashlib.sha256(payload).digest()
        header=dict(channel=channel,start=start,end=end,width=width)
        key=digest+json.dumps(header,sort_keys=True,allow_nan=False).encode()
        with self.lock:
            def compute():
                try:
                    if self.process is not None and self.process.poll() is not None:self.close()
                    if self.process is None:self._start()
                    raw=payload if self.digest!=digest else b''
                    request=json.dumps(dict(size=len(raw),**header)).encode()+b'\n'+raw
                    process=self.process;ready=queue.Queue(maxsize=1)
                    def exchange():
                        try:
                            process.stdin.write(request);process.stdin.flush()
                            size=int(process.stdout.readline(32))
                            if not 0<size<=512_000:raise ValueError('Preview output budget')
                            encoded=process.stdout.read(size)
                            if len(encoded)!=size:raise ValueError('Short preview response')
                            ready.put(json.loads(encoded))
                        except (OSError,ValueError):ready.put({'error':'preview_resource_failed'})
                    threading.Thread(target=exchange,daemon=True).start()
                    value=ready.get(timeout=timeout)
                    if 'error' in value:raise PreviewError(value['error'])
                    self.digest=digest
                    return value
                except Exception as exc:
                    self.close()
                    if isinstance(exc,PreviewError):raise
                    raise PreviewError('preview_timeout' if isinstance(exc,queue.Empty) else 'preview_resource_failed',503) from None
            value=self.cache.get(key,compute);self.touched=time.monotonic();return value

    def _schedule(self,delay):
        self.timer=threading.Timer(delay,self._expire);self.timer.daemon=True;self.timer.start()

    def _expire(self):
        with self.lock:
            remaining=self.idle_seconds-(time.monotonic()-self.touched)
            if self.process is None:return
            if remaining<=0:self.close()
            else:self._schedule(remaining)

    def close(self):
        with self.lock:
            if self.timer:self.timer.cancel();self.timer=None
            if self.stop:self.stop.set();self.stop=None
            if self.monitor:self.monitor.join(timeout=2);self.monitor=None
            if self.process:
                process,self.process=self.process,None;_close_group(process)
            self.digest=None;self.cache.clear()
