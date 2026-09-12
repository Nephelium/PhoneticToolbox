"""One explicit output device for live phonation, previews, and test tones."""
import json
import threading
import time
import numpy as np
import sounddevice as sd
from pathlib import Path
from phonetic_core.vocal_tract.monitor import OutputHistory
from phonetic_core.vocal_tract.envelope import attack_envelope


def audition_samples(samples,volume,gain_db,envelope=None):
    output=np.tanh(samples*4*volume*10**(gain_db/20))*.9
    return output if envelope is None else output*envelope

def output_devices():
    hosts=sd.query_hostapis(); devices=sd.query_devices()
    preferred=[i for i,d in enumerate(devices) if d['max_output_channels']>0 and hosts[d['hostapi']]['name']=='Windows WASAPI']
    if not preferred:preferred=[i for i,d in enumerate(devices) if d['max_output_channels']>0 and d['hostapi']==sd.default.hostapi]
    return [{'id':hosts[devices[i]['hostapi']]['name']+'|'+devices[i]['name'],'index':i,'name':devices[i]['name'],'channels':min(2,devices[i]['max_output_channels'])} for i in preferred]

class LiveAudio:
    def __init__(self,engine,*,playback_allowed=False,config_path=None):
        # Tests can explicitly prohibit all physical-device access.
        self.playback_allowed=playback_allowed
        self.engine=engine;self.params=engine.params.copy();self.f0=150.;self.volume=.8;self.lip_width=1.;self.tube=None
        self.source=None;self.gain_db=0.
        self.output_history=OutputHistory(engine.sr)
        devices=output_devices() if playback_allowed else []
        # User explicitly chose the built-in speakers on 2026-09-08.
        selected=next((d for d in devices if 'Realtek' in d['name']),devices[0] if devices else None)
        self.device_key=selected['id'] if selected else ''
        self.config=Path(config_path) if config_path else None
        if self.config and self.config.exists():
            saved=json.loads(self.config.read_text(encoding='utf-8'))
            self.device_key=saved.get('device',self.device_key);self.volume=float(np.clip(saved.get('volume',.8),0,1))
            self.gain_db=float(np.clip(saved.get('gain_db',0),0,24))
        self.active=False;self.thread=None;self.error=None;self.heartbeat=time.monotonic();self.mode='idle';self.buffer=None;self.envelope=None
        self.stats={};self.guard=threading.Lock();self.clock_start=None;self.completed=False;self.last_position=0.;self._reset_stats()

    def position(self):
        if self.completed and self.buffer is not None:return len(self.buffer)/self.engine.sr
        if not self.active:return self.last_position
        if self.clock_start is None:return 0.
        latency=(self.stats.get('device_latency_ms') or 0)/1000
        return max(0.,min(self.stats['generated_samples']/self.engine.sr,time.monotonic()-self.clock_start-latency))

    def _reset_stats(self):
        self.stats={'blocks':0,'underflows':0,'generated_samples':0,'compute_ms':0.,'device_latency_ms':None,'rms':0.,'peak':0.,'device':self.device_key.split('|')[-1]}

    def configure(self,device=None,volume=None,gain_db=None):
        if not self.playback_allowed:raise PermissionError('静音建模已锁定，禁止音频输出')
        # Validate the entire request before changing or stopping playback.
        if volume is not None and (not np.isfinite(volume) or not 0<=volume<=1):raise ValueError('Invalid volume')
        if gain_db is not None and (not np.isfinite(gain_db) or not 0<=gain_db<=24):raise ValueError('Invalid audition gain')
        if device is not None:
            if device not in [d['id'] for d in output_devices()]:raise ValueError('所选输出设备不可用，请重新选择')
            self.stop();self.device_key=device
        if volume is not None:
            self.volume=float(volume)
        if gain_db is not None:self.gain_db=float(gain_db)
        if self.config is None:return
        self.config.parent.mkdir(parents=True,exist_ok=True)
        self.config.write_text(json.dumps({'device':self.device_key,'volume':self.volume,'gain_db':self.gain_db},ensure_ascii=False,indent=2),encoding='utf-8')

    def start(self,buffer=None,mode='live',envelope=None):
        if not self.playback_allowed:raise PermissionError('静音建模已锁定，禁止音频输出')
        self.stop()
        with self.guard:
            if self.device_key not in [d['id'] for d in output_devices()]:raise ValueError('输出设备已断开，请选择其他设备')
            self.buffer=buffer;self.envelope=envelope;self.mode=mode;self.active=True;self.error=None;self.heartbeat=time.monotonic();self._reset_stats();self.clock_start=None;self.completed=False;self.last_position=0.
            self.output_history.reset()
            if buffer is None:self.tube=self.engine.prepare_tube(self.params,self.lip_width)
            self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def stop(self):
        self.last_position=self.position()
        self.active=False
        if self.thread and self.thread is not threading.current_thread():self.thread.join(timeout=3)
        if self.thread and self.thread.is_alive():raise RuntimeError('Audio thread did not stop')

    def _run(self):
        ended=False
        try:
            if not self.playback_allowed:raise PermissionError('Audio output is locked')
            device=next(d for d in output_devices() if d['id']==self.device_key)
            glottis=lambda:self.engine.glottis(self.f0,source=self.source) if self.source is not None else self.engine.glottis(self.f0)
            n=960;offset=0;g=glottis();tube=self.tube
            if self.buffer is None:
                with self.engine.lock:self.engine.reset_tube(tube,g)
            with sd.OutputStream(samplerate=self.engine.sr,channels=device['channels'],dtype='float32',blocksize=n,latency=.04,device=device['index']) as stream:
                self.stats['device_latency_ms']=stream.latency*1000;self.stats['device']=device['name']
                while self.active and time.monotonic()-self.heartbeat<5:
                    t=time.perf_counter()
                    envelope=None
                    if self.buffer is None:
                        target=self.tube;g=glottis()
                        tube={k:(target[k] if k=='articulators' else v+.4*(target[k]-v)) for k,v in tube.items()}
                        with self.engine.lock:block=self.engine.block_tube(tube,g,n)
                        block=block*10**((self.source or {}).get('audition_gain_db',0)/20)
                        envelope=attack_envelope(self.stats['generated_samples'],n,self.engine.sr)
                    else:
                        if offset>=len(self.buffer):ended=True;break
                        if self.envelope is not None:
                            envelope=self.envelope[offset:offset+n]
                            if len(envelope)<n:envelope=np.pad(envelope,(0,n-len(envelope)))
                        block=self.buffer[offset:offset+n];offset+=len(block)
                        if len(block)<n:block=np.pad(block,(0,n-len(block)))
                    self.stats['compute_ms']=(time.perf_counter()-t)*1000
                    # The same fixed audition gain is used on every route.
                    block=audition_samples(block,self.volume,self.gain_db,envelope)
                    self.stats['rms']=float(np.sqrt(np.mean(block*block)));self.stats['peak']=float(np.max(np.abs(block)))
                    stereo=np.repeat(block.astype(np.float32)[:,None],device['channels'],axis=1)
                    if self.clock_start is None:self.clock_start=time.monotonic()
                    self.stats['underflows']+=int(stream.write(stereo));self.stats['blocks']+=1;self.stats['generated_samples']+=n
                    self.output_history.append(block)
                # Short release avoids clicks on a user stop.
                if self.stats['blocks'] and not ended:
                    tail=block*np.linspace(1,0,n)
                    stream.write(np.repeat(tail.astype(np.float32)[:,None],device['channels'],axis=1))
                    self.output_history.append(tail)
        except Exception as exc:self.error=str(exc)
        finally:self.completed=ended and not self.error;self.last_position=self.position();self.active=False;self.mode='idle';self.stats['rms']=0.;self.stats['peak']=0.

    def test_tone(self):
        if not self.playback_allowed:raise PermissionError('静音建模已锁定，禁止音频输出')
        t=np.arange(int(self.engine.sr*.5))/self.engine.sr
        audio=.035*np.sin(2*np.pi*440*t)*np.minimum(1,t/.02)*np.minimum(1,(.5-t)/.02)
        self.start(audio,mode='test')
