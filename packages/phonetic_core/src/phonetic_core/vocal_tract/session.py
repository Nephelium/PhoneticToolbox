"""Shared stateful VTL operations; device, persistence and process ports are injected."""
import bisect
import copy
import base64
import hashlib
import json
import threading
import time
import zlib
import numpy as np
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.monitor import analyze_output
from phonetic_core.vocal_tract.source import validate_source
from phonetic_core.vocal_tract.perturbation import validate_perturbation
from phonetic_core.vocal_tract.shape import validate_tongue_shape
from phonetic_core.vocal_tract.trajectory import validate_frames, validate_pitch_curve, validate_pressure_curve, MAX_SEQUENCE_SECONDS
from phonetic_core.vocal_tract.animation import prepare_animation
from phonetic_core.vocal_tract.document import sequence_document, make_document
from .suprasegmentals import validate_features
from .lips import validate_lip_shape
from .envelope import audition_samples
from .sequence_controls import validate_global_controls, ipa_sequence
from .preset_pack import make_pack, validate_pack
from .preset_library import edit_library, active_set, validate_presets, name
import uuid


class Runtime:
    def __init__(self, engine, profile, live, *, output_devices=lambda: []):
        self.engine=engine;self.profile=profile;self.live=live;self.output_devices=output_devices;self.closed=False
        self.revision=0;self.section=self.engine.section_count//2;self.animation=None;self.animation_id=0;self.animation_playing=False
        self.cancel=threading.Event();self.manual_root=False
        self.cancel_generation=0;self.render_lock=threading.Lock()
        self.animation_keys=set();self.prepare_count=0
        self.monitor_lock=threading.Lock();self.monitor_cache=None
        self.perturbation=None

    def state(self):
        a=self.live
        return {'audio_locked':not a.playback_allowed,'active':a.active,'mode':a.mode,'audio_error':a.error,
                'audio':dict(a.stats),'revision':self.revision,'device':a.device_key,'volume':a.volume,'gain_db':a.gain_db}

    def invoke(self,path,obj=None,*,generation=None):
        obj={} if obj is None else obj
        if not isinstance(obj,dict):raise ValueError('无效的模块请求')
        if 'audition_auto_gain' in obj:
            if type(obj['audition_auto_gain']) is not bool:raise ValueError('试听自动增益须为开关值')
            self.live.auto_gain=obj['audition_auto_gain']
            obj={key:value for key,value in obj.items() if key!='audition_auto_gain'}
        self.live.heartbeat=time.monotonic()
        if path=='meta':
            return {**self.engine.metadata(),'api_version':'m10/1','source_updates':True,'audio_locked':not self.live.playback_allowed,
                'speaker_options':[{'id':'JD2','label':'男性'},{'id':'W02','label':'女性'}],
                'default_f0':getattr(self.engine,'default_f0',120.),
                'max_animation_duration':MAX_SEQUENCE_SECONDS,
                'output_devices':self.output_devices() if self.live.playback_allowed else [],
                'audio_settings':{'device':self.live.device_key,'volume':self.live.volume,'gain_db':self.live.gain_db}}
        if path in ('status','heartbeat'):return self.state()
        if path=='keyframes/load':return self.profile.load_sequence(self.engine)
        if path=='document/validate':return sequence_document(self.engine,obj)
        if path=='document/create':return make_document(self.engine,obj['frames'],obj.get('pitch_curve',[]),obj.get('pressure_curve',[]),obj.get('perturbation'),obj.get('global_controls'),obj.get('features'))
        if path=='speaker/select':return self.select_speaker(obj.get('speaker'))
        if path=='sequence/ipa':return ipa_sequence(self.engine,obj.get('text'),obj.get('presets'),obj.get('duration'),stop_timing=obj.get('stop_timing',False))
        if path=='preset-pack/create':return make_pack(self.engine,obj)
        if path=='preset-pack/validate':return validate_pack(self.engine,obj)
        if path=='source/perturbation':
            value=validate_perturbation(obj.get('perturbation'))
            self.engine.configure_perturbation(value)
            self.perturbation=value
            return {'perturbation':value}
        if path=='library/load':return self.profile.load_library(self.engine)
        if path=='library/save':return self.profile.save_library(self.engine,obj)
        if path=='library/edit':return self.profile.save_library(self.engine,edit_library(self.engine,self.profile.load_library(self.engine),obj))
        if path=='library/import':
            pack=validate_pack(self.engine,obj.get('document'))
            if pack.get('geometry_notice') and obj.get('recompute') is not True:raise ValueError('请确认按当前几何重算后导入')
            library=self.profile.load_library(self.engine)
            target=obj.get('target','new')
            if target=='new':library=edit_library(self.engine,library,dict(action='create',name=obj.get('name',pack['title'])))
            elif target!='current':raise ValueError('请选择导入目标套装')
            conflict=obj.get('conflict','keep')
            if conflict not in ('keep','replace'):raise ValueError('请选择同名构型处理方式')
            current=active_set(library);added=replaced=kept=0
            for pose in pack['presets']:
                old=next((p for p in current['presets'] if name(p['name'])==name(pose['name'])),None)
                if old is not None:
                    if conflict=='keep':kept+=1;continue
                    current['presets'][current['presets'].index(old)]={**pose,'id':old['id']};replaced+=1
                else:current['presets'].append({**pose,'id':uuid.uuid4().hex});added+=1
            current['presets']=validate_presets(self.engine,current['presets'],unique=True)
            result=self.profile.save_library(self.engine,library)
            return dict(library=result,added=added,replaced=replaced,kept=kept)
        if path=='presets/load':return self.profile.load_presets(self.engine)
        if path=='presets/save':return self.profile.save_presets(self.engine,obj.get('presets'))
        if path=='favorites/load':return self.profile.load_favorites()
        if path=='favorites/save':return self.profile.save_favorites(self.engine,obj.get('ids'))
        if path=='keyframes':
            frames=validate_frames(self.engine,obj['frames'],for_storage=True)
            curve=validate_pitch_curve(obj.get('pitch_curve',[]));pressure=validate_pressure_curve(obj.get('pressure_curve',[]))
            perturbation=validate_perturbation(obj.get('perturbation'))
            controls=validate_global_controls(self.engine,obj.get('global_controls'));features=validate_features(self.engine,obj.get('features'))
            if features:self.profile.save_frames(frames,curve,pressure,perturbation,controls,features)
            elif controls:self.profile.save_frames(frames,curve,pressure,perturbation,controls)
            elif perturbation is not None:self.profile.save_frames(frames,curve,pressure,perturbation)
            elif pressure:self.profile.save_frames(frames,curve,pressure)
            else:self.profile.save_frames(frames,curve)
            if not frames:
                self.live.stop()
                self.animation_playing=False
                self.animation=None
                self.animation_keys.clear()
                self.animation_id+=1
            return {'saved':len(frames)}
        if path=='audio/monitor':
            seconds=float(obj.get('seconds',3));window=float(obj.get('window_ms',20));hop=float(obj.get('hop_ms',10))
            if not np.isfinite([seconds,window,hop]).all() or not 1<=seconds<=12:raise ValueError('无效的分析范围')
            samples,end,generation=self.live.output_history.snapshot(seconds)
            settings=dict(seconds=seconds,window_ms=window,hop_ms=hop,end=end,generation=generation,
                          view_seconds=obj.get('view_seconds'),view_lag=obj.get('view_lag',0.),waveform_width=obj.get('waveform_width',700))
            key=(self.engine.sr,*settings.values())
            with self.monitor_lock:
                if self.monitor_cache is not None and self.monitor_cache[0]==key:return copy.deepcopy(self.monitor_cache[1])
                result=analyze_output(samples,self.engine.sr,**settings)
                self.monitor_cache=(key,result)
                return copy.deepcopy(result)
        if path=='audio/settings':self.live.configure(obj.get('device'),obj.get('volume'),obj.get('gain_db'));return self.state()
        if path=='audio/test':self.live.test_tone();return self.state()
        if path=='source/update':
            rev=obj['revision']
            if type(rev) is not int or rev<=self.revision:raise ValueError('过期的声源请求')
            f0=float(obj.get('f0',self.live.f0));self.engine.glottis(f0)
            source=validate_source(obj.get('source',self.live.source))
            if source['vibration']==0:self.live.pitch_gain=1.
            elif self.live.tube is not None:
                self.live.pitch_gain=self.engine.pitch_level_gain(self.live.tube,f0,source)
            self.live.f0=f0;self.live.source=source;self.revision=rev
            return {'revision':rev,'f0':f0,'source':source}
        if path=='pose':
            p=self.engine.validated(obj['params']);rev=obj['revision']
            if type(rev) is not int or rev<=self.revision:raise ValueError('过期的构形请求')
            count=obj.get('section_count',self.engine.section_count)
            if type(count) is not int or count not in self.engine.section_options:
                raise ValueError('声道采样截面数应为 '+ '、'.join(map(str,self.engine.section_options)))
            f0=float(obj.get('f0',self.live.f0));self.engine.glottis(f0)
            source=validate_source(obj.get('source'));width=float(obj.get('lip_width',1))
            larynx=self.engine.validated_larynx(obj.get('larynx_height',0.))
            shape=validate_tongue_shape(obj.get('tongue_shape'));lip=validate_lip_shape(obj.get('lip_shape'))
            manual=obj.get('manual_root',False)
            if type(manual) is not bool:raise ValueError('无效的舌根模式')
            if count!=self.engine.section_count:
                self.engine.set_section_count(count)
                self.animation=None;self.animation_keys.clear();self.animation_id+=1
            self.engine.set_manual_root(manual)
            notice=None
            if obj.get('keep_vowel',False):p,notice=self.engine.constrain_nasal_opening(p,width,larynx,shape,lip)
            state=self.engine.snapshot(p,int(obj.get('section',self.section)),width,larynx,shape,lip)
            state.update(nasal_constraint=notice,source=source,revision=rev,manual_root=manual)
            self.live.tube=self.engine.snapshot_tube
            if source['vibration']==0:self.live.pitch_gain=1.
            else:self.live.pitch_gain=self.engine.pitch_level_gain(self.live.tube,f0,source)
            self.revision=rev;self.section=state['section'];self.manual_root=manual
            self.live.params=p;self.live.f0=f0;self.live.lip_width=width;self.live.source=source
            self.live.larynx_height=larynx;self.live.tongue_shape=shape;self.live.lip_shape=lip
            return state
        if path in ('animation/stop','deactivate') or (path=='live' and not obj.get('active')):
            with self.render_lock:
                self.cancel_generation+=1;self.cancel.set()
            self.live.stop()
            pose=None
            if self.animation_playing and self.animation is not None:
                _,_,pose=self.animation_frame()
                self.adopt_animation_pose(pose)
            self.animation_playing=False
            return {**self.state(),**({'pose':pose} if pose is not None else {})}
        if path=='live':
            self.engine.set_manual_root(self.manual_root)
            self.live.start();return self.state()
        if path=='preview':
            self.begin_render(generation);self.live.stop()
            p=self.engine.validated(obj.get('params',self.live.params));duration=float(obj.get('duration',1.2))
            if not .3<=duration<=5:raise ValueError('试听时长应为 0.3–5 秒')
            poses=[self.engine.presets[n] for n in ('a','i','u')] if obj.get('sequence')=='a-i-u' else [p.tolist(),p.tolist()]
            frames=[{'params':q,'lip_shape':validate_lip_shape(obj.get('lip_shape',self.live.lip_shape)),'tongue_shape':validate_tongue_shape(obj.get('tongue_shape',self.live.tongue_shape)),'larynx_height':self.engine.validated_larynx(obj.get('larynx_height',self.live.larynx_height)),'lip_width':float(obj.get('lip_width',1)),'f0':self.live.f0,'source':self.live.source,
                'duration':duration/len(poses),'manual_root':self.manual_root} for q in poses]
            prepared=prepare_animation(self.engine,frames,pictures_enabled=False,cancel=self.cancel)
            if self.cancel.is_set():raise ValueError('已取消生成')
            self.live.start(prepared['audition_audio'],mode='preview',envelope=prepared['envelope']);return self.state()
        if path=='animation/prepare':
            return self.prepare_sequence(obj,generation)
        if path in ('animation/audio','animation/picture'):
            if self.animation is None or obj.get('id')!=self.animation_id:raise ValueError('动作序列已失效')
            if path=='animation/picture':
                index=obj.get('index')
                if type(index) is not int or not 0<=index<len(self.animation['pictures']):raise ValueError('无效的动作画面索引')
                return {'time':self.animation['times'][index],'state':json.loads(zlib.decompress(self.animation['pictures'][index]))}
            # Same audition transform as live playback; no device access during export.
            audio=audition_samples(self.animation['audition_audio'],self.live.volume,self.live.gain_db,self.animation['envelope']).astype('<f4')
            return {'sample_rate':self.engine.sr,'samples':len(audio),'base64':base64.b64encode(audio.tobytes()).decode('ascii')}
        if path in ('animation/play','animation/frame'):
            if self.animation is None or obj.get('id')!=self.animation_id:raise ValueError('动作序列已失效，请重新生成')
            if path=='animation/play':
                if self.cancel.is_set():raise ValueError('动作已取消')
                self.animation_playing=True;self.live.start(self.animation['audition_audio'],mode='animation',envelope=self.animation['envelope']);return self.state()
            index,seconds,state=self.animation_frame()
            if self.animation_playing and not self.live.active:
                self.adopt_animation_pose(state);self.animation_playing=False
            return {'index':index,'time':seconds,'duration':self.animation['duration'],'active':self.live.active,
                'completed':self.live.completed,'audio_error':self.live.error,'state':state}
        raise ValueError('不支持的声道操作')

    def animation_frame(self):
        seconds=min(self.animation['duration'],self.live.position())
        index=max(0,bisect.bisect_right(self.animation['times'],seconds)-1)
        return index,seconds,json.loads(zlib.decompress(self.animation['pictures'][index]))

    def adopt_animation_pose(self,state):
        self.live.params=self.engine.validated(state['params'])
        self.manual_root=state.get('manual_root',False)
        self.engine.set_manual_root(self.manual_root)
        self.live.lip_width=state['lip_width'];self.live.f0=state['f0'];self.live.source=state.get('source')
        self.live.larynx_height=state.get('larynx_height',0.)
        self.live.tongue_shape=list(state.get('tongue_shape',[0.]*4));self.live.lip_shape=list(state.get('lip_shape',[0.]*4))
        self.live.tube=self.engine.prepare_tube(self.live.params,self.live.lip_width,self.live.larynx_height,self.live.tongue_shape,self.live.lip_shape)
        self.live.pitch_gain=self.engine.pitch_level_gain(self.live.tube,self.live.f0,self.live.source)

    def prepare_sequence(self,obj,generation=None):
        self.begin_render(generation);self.live.stop();self.animation_playing=False
        original_manual=self.engine.manual_root
        try:
            if 'perturbation' in obj:
                value=validate_perturbation(obj['perturbation'])
                self.engine.configure_perturbation(value);self.perturbation=value
            frames=validate_frames(self.engine,obj['frames'])
            curve=validate_pitch_curve(obj.get('pitch_curve',[]))
            pressure=validate_pressure_curve(obj.get('pressure_curve',[]))
            controls=validate_global_controls(self.engine,obj.get('global_controls'));features=validate_features(self.engine,obj.get('features'))
            key=self.animation_key(frames,curve,bool(obj.get('keep_vowel')),pressure,controls,features)
            if self.animation is not None and key in self.animation_keys:return self.animation_info(True)
            if obj.get('keep_vowel',False):
                for f in frames:
                    if f.get('aspiration'):continue
                    self.engine.set_manual_root(f.get('manual_root',False))
                    f['params']=self.engine.constrain_nasal_opening(f['params'],f['lip_width'],f.get('larynx_height',0.),f.get('tongue_shape'),f.get('lip_shape'))[0].tolist()
            prepared=prepare_animation(self.engine,frames,self.section,pitch_curve=curve,pressure_curve=pressure,global_controls=controls,features=features,cancel=self.cancel)
            if self.cancel.is_set():raise ValueError('已取消生成')
            self.animation=prepared
            self.animation['pitch_curve']=curve;self.animation['pressure_curve']=pressure
            self.animation_keys={key,self.animation_key(prepared['frames'],curve,bool(obj.get('keep_vowel')),pressure,controls,features)}
            self.prepare_count+=1;self.animation_id+=1
            return self.animation_info(False)
        finally:
            self.engine.set_manual_root(original_manual)

    def animation_key(self,frames,curve,keep,pressure_curve=None,global_controls=None,features=None):
        obj={'features':features or [],'frames':frames,'curve':curve,'pressure_curve':pressure_curve or [],'global_controls':global_controls or {},'speaker':getattr(self.engine,'speaker','JD2'),'perturbation':self.perturbation,'keep':keep,'section':self.section,'section_count':self.engine.section_count,'version':'m10-r15-continuous','geometry_method':self.engine.geometry_method}
        return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    def select_speaker(self,speaker):
        if speaker not in ('JD2','W02'):raise ValueError('请选择男性或女性声道模型')
        if speaker==self.engine.speaker:return {'speaker':speaker}
        self.live.stop();self.cancel.set();self.cancel_generation+=1
        engine=self.engine;previous=engine.speaker;root=engine.resource_dir
        options=list(engine.section_options);count=engine.section_count
        engine.close()
        try:engine.__init__(speaker,resource_dir=root)
        except Exception:
            engine.__init__(previous,resource_dir=root)
            if 33 in options:engine.enable_web_sections()
            engine.section_options=options;engine.set_section_count(count)
            raise
        if 33 in options:engine.enable_web_sections()
        engine.section_options=options;engine.set_section_count(count)
        self.profile.set_speaker(speaker)
        self.live.params=engine.params.copy();self.live.f0=engine.default_f0
        self.live.lip_width=1.;self.live.larynx_height=0.;self.live.tongue_shape=[0.]*4;self.live.lip_shape=[0.]*4
        self.live.source=validate_source(None)
        self.live.pitch_gain=1.;self.live.tube=None
        self.manual_root=False;self.perturbation=None
        self.animation=None;self.animation_keys.clear();self.animation_id+=1
        self.monitor_cache=None
        return {'speaker':speaker}

    def animation_info(self,cached):
        a=self.animation
        return {'id':self.animation_id,'duration':a['duration'],'count':len(a['pictures']),
                'frames':a['frames'],'times':a['times'],'timeline':a['timeline'],'cached':cached,'prepare_count':self.prepare_count}

    def begin_render(self,generation):
        with self.render_lock:
            if generation is not None and generation!=self.cancel_generation:raise ValueError('已取消排队的生成')
            self.cancel=threading.Event()

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            self.cancel.set();self.live.playback_allowed=False;self.live.stop()
        finally:
            self.engine.close()
