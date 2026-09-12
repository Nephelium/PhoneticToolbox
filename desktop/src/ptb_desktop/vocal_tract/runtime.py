"""Stateful M10 operations in one application-owned native process; no HTTP."""
import bisect
import base64
import hashlib
import json
from pathlib import Path
import threading
import time
import zlib
import numpy as np
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.monitor import analyze_output
from phonetic_core.vocal_tract.source import validate_source
from phonetic_core.vocal_tract.trajectory import validate_frames, validate_pitch_curve
from phonetic_core.vocal_tract.animation import prepare_animation
from phonetic_core.vocal_tract.document import sequence_document, make_document
from .audio_output import LiveAudio, output_devices
from .profile import ProfileStore


class Runtime:
    def __init__(self, resource_dir, profile_dir, *, playback_allowed=True, legacy_profile=None):
        self.engine=Engine(resource_dir=resource_dir)
        self.profile=ProfileStore(profile_dir,legacy_directory=legacy_profile)
        self.live=LiveAudio(self.engine,playback_allowed=playback_allowed,config_path=Path(profile_dir)/'audio-settings.json')
        self.revision=0;self.section=65;self.animation=None;self.animation_id=0;self.animation_playing=False
        self.cancel=threading.Event();self.manual_root=False
        self.cancel_generation=0;self.render_lock=threading.Lock()
        self.animation_keys=set();self.prepare_count=0

    def state(self):
        a=self.live
        return {'audio_locked':not a.playback_allowed,'active':a.active,'mode':a.mode,'audio_error':a.error,
                'audio':dict(a.stats),'revision':self.revision,'device':a.device_key,'volume':a.volume,'gain_db':a.gain_db}

    def invoke(self,path,obj=None,*,generation=None):
        obj={} if obj is None else obj
        if not isinstance(obj,dict):raise ValueError('无效的模块请求')
        self.live.heartbeat=time.monotonic()
        if path=='meta':
            return {**self.engine.metadata(),'api_version':'m10/1','audio_locked':not self.live.playback_allowed,
                'output_devices':output_devices() if self.live.playback_allowed else [],
                'audio_settings':{'device':self.live.device_key,'volume':self.live.volume,'gain_db':self.live.gain_db}}
        if path in ('status','heartbeat'):return self.state()
        if path=='keyframes/load':return self.profile.load_sequence(self.engine)
        if path=='document/validate':return sequence_document(self.engine,obj)
        if path=='document/create':return make_document(self.engine,obj['frames'],obj.get('pitch_curve',[]))
        if path=='presets/load':return self.profile.load_presets(self.engine)
        if path=='presets/save':return self.profile.save_presets(self.engine,obj.get('presets'))
        if path=='keyframes':
            frames=validate_frames(self.engine,obj['frames'],for_storage=True)
            self.profile.save_frames(frames,validate_pitch_curve(obj.get('pitch_curve',[])))
            return {'saved':len(frames)}
        if path=='audio/monitor':
            seconds=float(obj.get('seconds',3));window=float(obj.get('window_ms',20));hop=float(obj.get('hop_ms',10))
            if not np.isfinite([seconds,window,hop]).all() or not 1<=seconds<=12:raise ValueError('无效的分析范围')
            samples,end,generation=self.live.output_history.snapshot(seconds)
            return analyze_output(samples,self.engine.sr,seconds=seconds,window_ms=window,hop_ms=hop,end=end,generation=generation)
        if path=='audio/settings':self.live.configure(obj.get('device'),obj.get('volume'),obj.get('gain_db'));return self.state()
        if path=='audio/test':self.live.test_tone();return self.state()
        if path=='pose':
            p=self.engine.validated(obj['params']);rev=obj['revision']
            if type(rev) is not int or rev<=self.revision:raise ValueError('过期的构形请求')
            f0=float(obj.get('f0',150));self.engine.glottis(f0)
            source=validate_source(obj.get('source'));width=float(obj.get('lip_width',1))
            manual=obj.get('manual_root',False)
            if type(manual) is not bool:raise ValueError('无效的舌根模式')
            self.engine.set_manual_root(manual)
            notice=None
            if obj.get('keep_vowel',False):p,notice=self.engine.constrain_nasal_opening(p,width)
            state=self.engine.snapshot(p,int(obj.get('section',65)),width)
            state.update(nasal_constraint=notice,source=source,revision=rev,manual_root=manual)
            self.revision=rev;self.section=state['section'];self.manual_root=manual
            self.live.params=p;self.live.f0=f0;self.live.lip_width=width;self.live.source=source
            if self.live.active and self.live.mode=='live':self.live.tube=self.engine.prepare_tube(p,width)
            return state
        if path in ('animation/stop','deactivate') or (path=='live' and not obj.get('active')):
            with self.render_lock:
                self.cancel_generation+=1;self.cancel.set()
            self.live.stop();self.animation_playing=False
            return self.state()
        if path=='live':self.live.start();return self.state()
        if path=='preview':
            self.begin_render(generation);self.live.stop()
            p=self.engine.validated(obj.get('params',self.live.params));duration=float(obj.get('duration',1.2))
            if not .3<=duration<=5:raise ValueError('试听时长应为 0.3–5 秒')
            poses=[self.engine.presets[n] for n in ('a','i','u')] if obj.get('sequence')=='a-i-u' else [p.tolist(),p.tolist()]
            frames=[{'params':q,'lip_width':float(obj.get('lip_width',1)),'f0':self.live.f0,'source':self.live.source,
                'duration':duration/len(poses),'manual_root':self.manual_root} for q in poses]
            prepared=prepare_animation(self.engine,frames,pictures_enabled=False,cancel=self.cancel)
            if self.cancel.is_set():raise ValueError('已取消生成')
            self.live.start(prepared['audition_audio'],mode='preview',envelope=prepared['envelope']);return self.state()
        if path=='animation/prepare':
            self.begin_render(generation);self.live.stop();self.animation_playing=False
            frames=validate_frames(self.engine,obj['frames'])
            curve=validate_pitch_curve(obj.get('pitch_curve',[]))
            key=self.animation_key(frames,curve,bool(obj.get('keep_vowel')))
            if self.animation is not None and key in self.animation_keys:return self.animation_info(True)
            if obj.get('keep_vowel',False):
                for f in frames:
                    self.engine.set_manual_root(f.get('manual_root',False))
                    f['params']=self.engine.constrain_nasal_opening(f['params'],f['lip_width'])[0].tolist()
            prepared=prepare_animation(self.engine,frames,self.section,pitch_curve=curve,cancel=self.cancel)
            if self.cancel.is_set():raise ValueError('已取消生成')
            self.animation=prepared
            self.animation['pitch_curve']=curve
            self.animation_keys={key,self.animation_key(prepared['frames'],curve,bool(obj.get('keep_vowel')))}
            self.prepare_count+=1
            self.animation_id+=1
            return self.animation_info(False)
        if path in ('animation/audio','animation/picture'):
            if self.animation is None or obj.get('id')!=self.animation_id:raise ValueError('动作序列已失效')
            if path=='animation/picture':
                index=obj.get('index')
                if type(index) is not int or not 0<=index<len(self.animation['pictures']):raise ValueError('无效的动作画面索引')
                return {'time':self.animation['times'][index],'state':json.loads(zlib.decompress(self.animation['pictures'][index]))}
            # Same audition transform as live playback; no device access during export.
            from .audio_output import audition_samples
            audio=audition_samples(self.animation['audition_audio'],self.live.volume,self.live.gain_db,self.animation['envelope']).astype('<f4')
            return {'sample_rate':self.engine.sr,'samples':len(audio),'base64':base64.b64encode(audio.tobytes()).decode('ascii')}
        if path in ('animation/play','animation/frame'):
            if self.animation is None or obj.get('id')!=self.animation_id:raise ValueError('动作序列已失效，请重新生成')
            if path=='animation/play':
                if self.cancel.is_set():raise ValueError('动作已取消')
                self.animation_playing=True;self.live.start(self.animation['audition_audio'],mode='animation',envelope=self.animation['envelope']);return self.state()
            seconds=min(self.animation['duration'],self.live.position())
            index=max(0,bisect.bisect_right(self.animation['times'],seconds)-1)
            state=json.loads(zlib.decompress(self.animation['pictures'][index]))
            if self.animation_playing and not self.live.active:
                self.animation_playing=False;self.live.params=self.engine.validated(state['params'])
                self.live.lip_width=state['lip_width'];self.live.f0=state['f0'];self.live.source=state.get('source')
            return {'index':index,'time':seconds,'duration':self.animation['duration'],'active':self.live.active,
                'completed':self.live.completed,'audio_error':self.live.error,'state':state}
        raise ValueError('不支持的声道操作')

    def animation_key(self,frames,curve,keep):
        obj={'frames':frames,'curve':curve,'keep':keep,'section':self.section,'version':'m10-r5'}
        return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    def animation_info(self,cached):
        a=self.animation
        return {'id':self.animation_id,'duration':a['duration'],'count':len(a['pictures']),
                'frames':a['frames'],'times':a['times'],'cached':cached,'prepare_count':self.prepare_count}

    def begin_render(self,generation):
        with self.render_lock:
            if generation is not None and generation!=self.cancel_generation:raise ValueError('已取消排队的生成')
            self.cancel=threading.Event()

    def close(self):
        self.cancel.set();self.live.playback_allowed=False;self.live.stop();self.engine.close()
