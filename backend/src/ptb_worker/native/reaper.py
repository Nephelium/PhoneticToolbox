"""M01-C source_id SRC-REAPER: bounded named-pipe output, no arbitrary output path."""
import hashlib
import math
import re
import time
from pathlib import Path
from ..io.limits import LimitedBuffer,Limits,LimitError,Cancelled,FormatError
from ..io.scratch import Scratch,no_links
import sys

REAPER_SHA256='279fecc82ed0a49b0277b114270771d7670299068e849058b392672825981824'


def collect_pipe(argv,pipe,cwd,limits,stop=lambda:False,on_started=None,on_chunk=None,evidence=None):
    from .windows import OwnedProcess
    process=None;buffer=LimitedBuffer(limits.output_bytes);started=time.monotonic();received=0
    try:
        if stop():raise Cancelled('cancelled')
        process=OwnedProcess(argv,cwd,limits.process_bytes)
        if on_started:on_started(process.pid)
        while True:
            if stop():raise Cancelled('cancelled')
            if time.monotonic()-started>limits.timeout_seconds:raise LimitError('native_timeout')
            chunk,eof=pipe.read(min(4096,limits.output_bytes-received+1),process)
            if chunk:
                received+=len(chunk)
                if received>limits.output_bytes:raise LimitError("output_bytes_exceeded")
                if on_chunk:on_chunk(chunk)
                else:buffer.write(chunk)
            elif process.poll() is not None:
                if process.poll()!=0:raise FormatError('native_exit_'+str(process.poll()))
                return buffer.getvalue(),process.pid
            # A ready pipe must drain at pipe speed. Sleeping after every 4 KiB
            # throttled a 77 s EGG preview WAV for tens of seconds. Keep the
            # cancellation/deadline/budget checks above on every iteration.
            if not chunk: time.sleep(.005)
    finally:
        try:
            if process and evidence is not None:
                try:evidence.update(main_pid=process.pid,memory_peak_bytes=process.memory_peak(),configured_memory_bytes=limits.process_bytes,elapsed_seconds=time.monotonic()-started)
                except OSError:evidence['memory_probe_failed']=True
        finally:
            try:
                if process:process.close()
                if process and evidence is not None:evidence['cleaned']=getattr(process,'group_cleaned',False)
            finally:pipe.close()


class Reaper:
    def __init__(self,binary,scratch,limits=Limits(),stop=lambda:False,on_started=None,on_chunk=None):
        if not isinstance(scratch,Scratch):raise TypeError('Host-owned Scratch capability required')
        self.binary=Path(binary).absolute();no_links(self.binary)
        self.sha256=REAPER_SHA256
        if sys.platform=='linux':
            from .linux_runtime import load_profile
            _,profile=load_profile()
            if str(self.binary)!=profile.get('reaper_binary'):
                raise ValueError('Unregistered native binary')
            self.sha256=profile['hashes'].get(str(self.binary))
        if not self.sha256 or self.binary.stat().st_size>16_000_000 or hashlib.sha256(self.binary.read_bytes()).hexdigest()!=self.sha256:
            raise ValueError('Unregistered native binary')
        self.scratch=scratch;self.limits=limits;self.stop=stop;self.on_started=on_started

    def __call__(self,audio,frame_interval_sec,min_f0,max_f0,*,hilbert,no_highpass):
        # The orchestration worker also uses collect_pipe. Scientific DLLs must
        # load only when executing REAPER inside the owned scientific child.
        from scipy.io import wavfile
        import numpy as np
        from phonetic_core.acoustic.reaper_codec import reaper_pcm16,parse_est_f0
        from phonetic_core.ports.acoustic import ReaperTrack
        if self.stop():raise Cancelled('cancelled')
        if audio.samples.size>self.limits.samples:raise LimitError('sample_limit_exceeded')
        converted_frames=math.ceil(len(audio.samples)*16000/audio.sample_rate_hz)
        if converted_frames>self.limits.samples or converted_frames*2+44>self.limits.input_bytes:
            raise LimitError('converted_sample_limit_exceeded')
        if not all(math.isfinite(x) for x in [frame_interval_sec,min_f0,max_f0]) or not (0<frame_interval_sec and 0<min_f0<max_f0<=8000):
            raise ValueError('Invalid REAPER parameters')
        pcm=reaper_pcm16(audio)
        stream=LimitedBuffer(self.limits.input_bytes);wavfile.write(stream,16000,pcm)
        path=self.scratch.create(stream.getvalue(),'.wav')
        pipe=None
        try:
            if sys.platform!='linux':
                from .windows import InputPipe
                pipe=InputPipe()
            argv=[str(self.binary),'-i',str(path),'-f',pipe.name if pipe else '', '-a','-e',str(float(frame_interval_sec)),
                  '-m',str(float(min_f0)),'-x',str(float(max_f0))]
            if hilbert:argv.append('-t')
            if no_highpass:argv.append('-s')
            if sys.platform=='linux':
                from .linux_reaper import collect
                payload,self.last_pid=collect(argv,self.scratch.root,self.limits,self.stop,self.on_started)
            else:
                payload,self.last_pid=collect_pipe(argv,pipe,self.scratch.root,self.limits,self.stop,self.on_started)
            self.last_output_bytes=len(payload)
            try:times,voiced,values=parse_est_f0(payload.decode('ascii'))
            except UnicodeError as exc:raise FormatError('invalid_native_encoding') from exc
            count=re.search(rb'^NumFrames ([0-9]+)\r?$',payload,re.MULTILINE)
            if (not len(times) or not payload.startswith(b'EST_File Track\n') or count is None or
                int(count.group(1))!=len(times) or not np.isfinite(times).all() or not np.isfinite(values).all() or
                np.any(np.diff(times)<=0) or not np.isin(voiced,[0,1]).all()):
                raise FormatError('invalid_native_track')
            return ReaperTrack(times,np.where(values>0.,values,np.nan),'native_reaper')
        finally:
            if pipe:pipe.close()
            self.scratch.remove(path)
