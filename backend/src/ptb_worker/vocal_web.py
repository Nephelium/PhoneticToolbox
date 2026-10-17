"""Device-free ports for the shared VTL session. Lives only in its owned worker."""
import base64
import json
import math
import time
import threading
from functools import wraps
import numpy as np
from phonetic_core.vocal_tract.envelope import audition_samples, attack_envelope
from phonetic_core.vocal_tract.monitor import OutputHistory
from phonetic_core.vocal_tract.trajectory import validate_frames, validate_pitch_curve

WEB_SEQUENCE_SECONDS=5.


def validate_web_operation(runtime,op,body):
    """Server policy for bounded browser audition; desktop keeps its own policy."""
    if op=='live' and body.get('active'):
        raise ValueError('由于服务器性能限制，网页版暂不支持持续发声，请下载桌面版。')
    if runtime.live.active and runtime.live.mode=='preview' and op in ('pose','preview','audio/test','audio/settings','animation/prepare','animation/play'):
        raise ValueError('请等待测试发声结束后再更改构形或开始合成。')
    if op=='preview' and (type(body.get('duration')) not in (int,float) or body['duration']!=.5 or body.get('sequence') is not None):
        raise ValueError('网页版仅支持当前构形的 0.5 秒测试发声。')
    if op=='animation/prepare':
        frames=validate_frames(runtime.engine,body.get('frames'))
        if round(sum(f['duration'] for f in frames),9)>WEB_SEQUENCE_SECONDS:
            raise ValueError('网页版单次关键帧播放和视频导出最多 5 秒，请缩短序列或下载桌面版。')
    if op=='video/begin' and float(body.get('duration',0))>WEB_SEQUENCE_SECONDS:
        raise ValueError('网页版单次视频导出最多 5 秒，请缩短序列或下载桌面版。')


class MemoryProfile:
    def __init__(self):
        self.sequence={'frames':[], 'pitch_curve':[]};self.presets=[]
        self.favorites=['a','i','u','e','o','E','y','2','l','n','t']

    def load_sequence(self,engine):return self.sequence
    def save_frames(self,frames,pitch_curve):self.sequence={'frames':frames,'pitch_curve':validate_pitch_curve(pitch_curve)}
    def load_presets(self,engine):return self.presets
    def save_presets(self,engine,presets):
        if not isinstance(presets,list):raise ValueError('无效的构形库')
        result=[];ids=set()
        for item in presets:
            if not isinstance(item,dict):raise ValueError('无效的构形')
            key=item.get('id')
            if not isinstance(key,str) or not 1<=len(key)<=64 or key in ids or key in engine.presets:raise ValueError('构形标识重复或无效')
            pose=validate_frames(engine,[item],for_storage=True)[0]
            if pose.get('aspiration') or not pose['name'].strip():raise ValueError('请保存有名称的构形')
            ids.add(key);result.append({**pose,'id':key})
        if len(json.dumps(result,ensure_ascii=False).encode())>8_000_000:raise ValueError('构形库超过容量')
        self.presets=result;return result
    def load_favorites(self):return self.favorites
    def save_favorites(self,engine,ids):
        available=set(engine.presets)|{p['id'] for p in self.presets}
        if not isinstance(ids,list) or len(ids)>500 or any(not isinstance(k,str) or k not in available for k in ids) or len(set(ids))!=len(ids):raise ValueError('无效的常用构形设置')
        self.favorites=list(ids);return self.favorites


def locked(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        with self.guard:return method(self,*args,**kwargs)
    return call


class BrowserAudio:
    """Pull PCM in the worker; only the browser opens an output device.

    Uses the desktop's 960-sample smoothing, source gain, audition transform and
    attack. History is committed at the browser's acknowledged playback position.
    """
    def __init__(self,engine):
        self.guard=threading.RLock()
        self.engine=engine;self.playback_allowed=True;self.params=engine.params.copy()
        self.f0=150.;self.volume=.8;self.gain_db=0.;self.lip_width=1.;self.larynx_height=0.;self.tongue_shape=[0.]*4;self.source=None
        self.device_key='browser';self.active=False;self.mode='idle';self.error=None;self.completed=False;self.heartbeat=time.monotonic()
        self.buffer=None;self.envelope=None;self.tube=None;self.position_seconds=0.;self.offset=0;self.committed=0
        self.pending=np.empty(0,dtype=np.float32);self.output_history=OutputHistory(engine.sr)
        self.stats={'blocks':0,'underflows':0,'generated_samples':0,'compute_ms':0.,'device_latency_ms':None,'rms':0.,'peak':0.,'device':'浏览器音频输出'}

    def position(self):return self.position_seconds
    @locked
    def configure(self,device=None,volume=None,gain_db=None):
        if device not in (None,'browser'):raise ValueError('请在浏览器中选择输出设备')
        for n,lo,hi in [(volume,0,1),(gain_db,0,24)]:
            if n is not None and (type(n) not in (int,float) or not math.isfinite(n) or not lo<=n<=hi):raise ValueError('无效的音量或增益')
        if volume is not None:self.volume=float(volume)
        if gain_db is not None:self.gain_db=float(gain_db)

    @locked
    def start(self,buffer=None,mode='live',envelope=None):
        self.stop();self.active=True;self.mode=mode;self.completed=False;self.error=None
        self.buffer=buffer;self.envelope=envelope;self.offset=self.committed=0;self.position_seconds=0.
        self.pending=np.empty(0,dtype=np.float32);self.output_history.reset();self.stats.update(blocks=0,generated_samples=0,rms=0.,peak=0.)
        if buffer is None:
            self.tube=self.engine.prepare_tube(self.params,self.lip_width,self.larynx_height,self.tongue_shape)
            self.current_tube=self.tube;self.engine.reset_tube(self.tube,self.engine.glottis(self.f0,source=self.source))

    @locked
    def stop(self):
        self.active=False;self.mode='idle';self.stats.update(rms=0.,peak=0.)

    @locked
    def acknowledge(self,obj):
        if obj.get('playback_generation',self.output_history.generation)!=self.output_history.generation:return
        value=obj.get('position',self.position_seconds)
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=self.offset/self.engine.sr+1e-6:raise ValueError('无效的播放位置')
        at=max(self.committed,min(self.offset,round(value*self.engine.sr)))
        n=at-self.committed
        if n:
            samples=self.pending[:n];self.output_history.append(samples);self.pending=self.pending[n:];self.committed=at
            self.stats['rms']=float(np.sqrt(np.mean(samples*samples)));self.stats['peak']=float(np.max(np.abs(samples)))
        self.position_seconds=at/self.engine.sr
        if obj.get('completed') is True:
            if self.buffer is None or self.committed<len(self.buffer):raise ValueError('音频尚未播放完整')
            self.completed=True;self.active=False;self.mode='idle';self.stats.update(rms=0.,peak=0.)

    @locked
    def pull(self,obj):
        if obj.get('playback_generation',self.output_history.generation)!=self.output_history.generation:return self.packet(np.empty(0,dtype=np.float32),True)
        self.acknowledge(obj)
        if not self.active:return self.packet(np.empty(0,dtype=np.float32),True)
        if self.offset-self.committed>self.engine.sr:raise ValueError('音频缓冲已满，请等待播放')
        begin=time.perf_counter();parts=[];n=960;start=self.offset
        # Prepared PCM needs enough lead to cover server/SSH round trips.
        # Keep the desktop's 960-sample transform and live control cadence.
        for _ in range(24 if self.buffer is not None else 8):
            envelope=None
            if self.buffer is None:
                target=self.tube;g=self.engine.glottis(self.f0,source=self.source)
                self.current_tube={k:(target[k] if k=='articulators' else v+.4*(target[k]-v)) for k,v in self.current_tube.items()}
                block=self.engine.block_tube(self.current_tube,g,n)*10**((self.source or {}).get('audition_gain_db',0)/20)
                envelope=attack_envelope(self.offset,n,self.engine.sr)
            else:
                block=self.buffer[self.offset:self.offset+n]
                if not len(block):break
                if self.envelope is not None:envelope=self.envelope[self.offset:self.offset+len(block)]
            block=audition_samples(block,self.volume,self.gain_db,envelope).astype('<f4')
            parts.append(block);self.offset+=len(block);self.stats['blocks']+=1
        audio=np.concatenate(parts) if parts else np.empty(0,dtype='<f4')
        self.pending=np.concatenate([self.pending,audio]);self.stats['generated_samples']=self.offset;self.stats['compute_ms']=(time.perf_counter()-begin)*1000
        return {**self.packet(audio,self.buffer is not None and self.offset>=len(self.buffer)),'offset':start}

    def packet(self,audio,eof):return {'sample_rate':self.engine.sr,'samples':len(audio),'base64':base64.b64encode(audio.tobytes()).decode(),'eof':eof,'playback_generation':self.output_history.generation}
    def test_tone(self):
        t=np.arange(int(self.engine.sr*.5))/self.engine.sr
        self.start(.035*np.sin(2*np.pi*440*t)*np.minimum(1,t/.02)*np.minimum(1,(.5-t)/.02),mode='test')
