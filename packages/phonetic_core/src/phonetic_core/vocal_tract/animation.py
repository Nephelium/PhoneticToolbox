"""Precompute synchronized audio and geometry in the worker."""
import json
import zlib
import numpy as np
from phonetic_core.vocal_tract.trajectory import validate_frames, sample_trajectory, validate_pitch_curve, validate_pressure_curve
from phonetic_core.vocal_tract.envelope import sequence_envelope
from .sequence_controls import validate_global_controls, apply_global_controls
from .suprasegmentals import validate_features, compile_features, apply_features
from .stop_timing import timing_events
from .perturbation import DEFAULT_PERTURBATION

def prepare_animation(engine,frames,section=None,*,pictures_enabled=True,pitch_curve=None,pressure_curve=None,global_controls=None,features=None,cancel=None):
    baseline=engine.perturbation;manual_root=engine.manual_root
    try:
        return _prepare_animation(engine,frames,section,pictures_enabled=pictures_enabled,pitch_curve=pitch_curve,
            pressure_curve=pressure_curve,global_controls=global_controls,features=features,cancel=cancel)
    finally:
        # A regional effect must not become the next live-preview setting,
        # including when rendering fails or is cancelled.
        try:
            if engine.perturbation!=baseline:engine.configure_perturbation(baseline)
        finally:
            engine.set_manual_root(manual_root)


def _prepare_animation(engine,frames,section=None,*,pictures_enabled=True,pitch_curve=None,pressure_curve=None,global_controls=None,features=None,cancel=None):
    curve=validate_pitch_curve(pitch_curve or [])
    pressure=validate_pressure_curve(pressure_curve or [])
    controls=validate_global_controls(engine,global_controls)
    frames=validate_frames(engine,frames)
    # Authoring remains in seconds; rendering shares the existing nearest-sample
    # convention with silence/stop boundaries, including non-grid-aligned edits.
    compiled=[(f,[(round(a*engine.sr)/engine.sr,round(b*engine.sr)/engine.sr) for a,b in intervals])
              for f,intervals in compile_features(validate_features(engine,features),frames)]
    n=round(sum(f['duration'] for f in frames)*engine.sr);duration=n/engine.sr;block=960
    total=duration
    regional_source=any(k in controls for k in ('jitter','shimmer')) or any(f['parameter'] in ('jitter','shimmer') for f,_ in compiled)
    if regional_source and not hasattr(engine.lib,'vtlSetSourcePerturbation'):
        raise ValueError('当前声源组件不支持 jitter/shimmer 超音段特征，请更新声源组件后重试。')
    baseline=engine.perturbation or DEFAULT_PERTURBATION
    def sample_frame(frames,seconds):
        pose=sample_trajectory(frames,seconds,curve,pressure)
        if regional_source:pose={**pose,'perturbation':dict(baseline)}
        return apply_features(engine,apply_global_controls(engine,pose,controls),compiled,seconds,total)
    def glottis(pose):
        if pose.get('silent'):return engine.glottis(150,source={'mode':'voiceless','pressure_pa':0,'vibration':0})
        return engine.glottis(pose['f0'],source=pose['source']) if 'source' in pose else engine.glottis(pose['f0'])
    frames=validate_frames(engine,frames)
    # Mixed automatic/manual endpoints use their actual root coordinates.
    if any(f.get('manual_root',False) for f in frames) or any(k in controls for k in ('TRX','TRY')) or any(f['parameter'] in ('TRX','TRY') for f,_ in compiled):
        for f in frames:
            if not f.get('aspiration') and not f['manual_root']:
                engine.set_manual_root(False)
                actual=engine.snapshot(f['params'],section,f['lip_width'],f.get('larynx_height',0.),f.get('tongue_shape'),f.get('lip_shape'))['limited']
                for key in ('TRX','TRY'):
                    i=engine.names.index(key);f['params'][i]=actual[i]
                f['manual_root']=True
    silent_intervals=[];source_intervals=[];stop_boundaries=set();elapsed=0.
    for f in frames:
        start=round(elapsed*engine.sr);elapsed+=f['duration']
        if f.get('silent'):silent_intervals.append((start,round(elapsed*engine.sr)))
        if f.get('aspiration'):source_intervals.append((start,round(elapsed*engine.sr)))
        if f.get('stop_timing'):stop_boundaries.update((start,round(elapsed*engine.sr)))
    events,fine=timing_events(frames)
    fine=[(round(a*engine.sr),round(b*engine.sr)) for a,b in fine]
    feature_events={t for f,intervals in compiled for interval in intervals for t in interval if 0<=t<=duration}
    feature_boundaries={round(t*engine.sr) for t in feature_events}
    events.update(feature_events)
    event_samples={round(t*engine.sr) for t in events if 0<=t<=duration}
    cuts=sorted({edge for interval in silent_intervals+source_intervals for edge in interval}|event_samples|{n})
    chunks=[];gains=[];pictures=[];times=[];first=sample_frame(frames,0)
    # Geometry is independent of F0 and source pressure. Reuse only an exactly
    # identical adjacent pose; moving poses still run the original model.
    def geometry_key(pose):
        return (tuple(pose['params']),pose['lip_width'],pose.get('larynx_height',0.),
                tuple(pose.get('tongue_shape') or [0.]*4),tuple(pose.get('lip_shape') or [0.]*4),pose.get('manual_root',False))
    # Reference gains are sampled at pose/curve boundaries and at most 20 Hz
    # pitch increments. This keeps interactive synthesis bounded; it never
    # follows actual shimmer or the pressure envelope.
    anchors={0.,duration}|{t for t in events if 0<=t<=duration};elapsed=0.
    # The reference gain must obey the same discontinuities as the source and
    # tract. Otherwise gain interpolation alone anticipates a regional F0 edit.
    anchors.update(float(np.nextafter(t,-np.inf)) for t in feature_events if t>0)
    for index,f in enumerate(frames):
        start=elapsed;elapsed+=f['duration'];anchors.add(elapsed)
        if index+1<len(frames) and not f.get('aspiration'):
            a=sample_frame(frames,start)['f0'];b=sample_frame(frames,elapsed)['f0']
            anchors.update(np.linspace(start,elapsed,max(1,int(np.ceil(abs(b-a)/20)))+1))
    for (a,fa),(b,fb) in zip(curve[:-1],curve[1:]):
        anchors.update(np.linspace(a*duration,b*duration,max(1,int(np.ceil(abs(fb-fa)/20)))+1))
    level_times=sorted(anchors);level_gains=[];reference_tubes={}
    for time in level_times:
        if cancel is not None and cancel.is_set():raise ValueError('已取消生成')
        pose=sample_frame(frames,time);key=geometry_key(pose)
        if pose.get('silent'):
            level_gains.append(1.);continue
        if key not in reference_tubes:
            engine.set_manual_root(pose.get('manual_root',False))
            reference_tubes[key]=engine.prepare_tube(pose['params'],pose['lip_width'],pose.get('larynx_height',0.),pose.get('tongue_shape'),pose.get('lip_shape'))
        level_gains.append(engine.pitch_level_gain(reference_tubes[key],pose['f0'],pose.get('source')))
    with engine.lock:
        engine.set_manual_root(first.get('manual_root',False))
        previous_key=geometry_key(first)
        previous_tube=reference_tubes.get(previous_key)
        if previous_tube is None:previous_tube=engine.prepare_tube(first['params'],first['lip_width'],first.get('larynx_height',0.),first.get('tongue_shape'),first.get('lip_shape'))
        if regional_source:engine.configure_perturbation(first['perturbation'])
        engine.reset_tube(previous_tube,glottis(first))
        offset=0;previous_silent=first.get('silent',False)
        while offset<n:
            if cancel is not None and cancel.is_set():raise ValueError('已取消生成')
            step=48 if any(a<=offset<b for a,b in fine) else block
            size=min(step,next(edge for edge in cuts if edge>offset)-offset)
            # Strength changes follow half-open intervals at the block START.
            # Updating strengths retains the seed and RNG progression.
            if regional_source:
                setting=sample_frame(frames,offset/engine.sr)['perturbation']
                if setting!=engine.perturbation:engine.configure_perturbation(setting)
            # Half-open intervals: the last sample of a voiced block belongs to
            # that block, even when its endpoint is the next silence boundary.
            endpoint=offset+size
            end_time=(endpoint-(.25 if silent_intervals or source_intervals or endpoint in stop_boundaries else 0))/engine.sr
            if endpoint in feature_boundaries:end_time=float(np.nextafter(endpoint/engine.sr,-np.inf))
            pose=sample_frame(frames,end_time)
            engine.set_manual_root(pose.get('manual_root',False))
            key=geometry_key(pose)
            if key!=previous_key:
                previous_tube=engine.prepare_tube(pose['params'],pose['lip_width'],pose.get('larynx_height',0.),pose.get('tongue_shape'),pose.get('lip_shape'));previous_key=key
            tube=previous_tube
            # Zero-length native update switches excitation at the exact boundary
            # without resetting the resonator state or smearing a 10 ms interval.
            if offset in stop_boundaries or offset in feature_boundaries or offset in {edge for interval in source_intervals for edge in interval}:
                boundary=sample_frame(frames,offset/engine.sr)
                engine.set_manual_root(boundary.get('manual_root',False))
                boundary_tube=engine.prepare_tube(boundary['params'],boundary['lip_width'],boundary.get('larynx_height',0.),boundary.get('tongue_shape'),boundary.get('lip_shape'))
                engine.block_tube(boundary_tube,glottis(boundary),0)
            if previous_silent and not pose.get('silent'):engine.reset_tube(tube,glottis(pose))
            chunks.append(engine.block_tube(tube,glottis(pose),size))
            level=np.exp(np.interp((offset+np.arange(size))/engine.sr,level_times,np.log(level_gains)))
            if (pose.get('source') or {}).get('vibration')==0:level=np.ones(size)
            gains.append(level*10**((pose.get('source') or {}).get('audition_gain_db',0)/20))
            previous_silent=pose.get('silent',False);offset+=size
            # Display frames are cached before playback. Geometry computation
            # cannot delay or perturb the native audio stream during playback.
        if pictures_enabled:
            shown_key=None;shown_geometry=None
            for index in range((n*30+engine.sr-1)//engine.sr+1):
                if cancel is not None and cancel.is_set():raise ValueError('已取消生成')
                time=min(index/30,duration);shown=sample_frame(frames,time)
                engine.set_manual_root(shown.get('manual_root',False))
                key=geometry_key(shown)
                if key!=shown_key:
                    shown_geometry=engine.snapshot(shown['params'],section,shown['lip_width'],shown.get('larynx_height',0.),shown.get('tongue_shape'),shown.get('lip_shape'));shown_key=key
                    state=dict(shown_geometry)
                else:state={**shown_geometry,'compute_ms':0.}
                state['f0']=shown['f0']
                state.update(source=shown.get('source'),manual_root=shown.get('manual_root',False))
                if regional_source:state['perturbation']=shown['perturbation']
                if controls:state['global_limited']=shown.get('global_limited',[])
                if shown.get('silent'):state['silent']=True
                if shown.get('aspiration'):state['aspiration']=True
                if shown.get('stop_phase'):state['stop_phase']=shown['stop_phase']
                pictures.append(zlib.compress(json.dumps(state,separators=(',',':')).encode(),1));times.append(time)
    # Scalar final controls share the exact sampler used for audio/pictures.
    # Include both sides of discontinuities so video never draws an invented
    # pitch ramp across a fixed feature, a silence or an unvoiced interval.
    knots={0.,duration,*np.linspace(0,duration,401),*events,*times};elapsed=0.
    for frame in frames:
        elapsed+=frame['duration'];knots.add(min(duration,elapsed))
    knots.update(p[0]*duration for p in curve)
    knots={float(t) for t in knots if 0<=t<=duration}
    knots.update(float(np.nextafter(t,-np.inf)) for t in list(knots) if t>0)
    timeline=[]
    for t in sorted(knots):
        if cancel is not None and cancel.is_set():raise ValueError('已取消生成')
        state=sample_frame(frames,t)
        voiced=not state.get('silent') and not state.get('aspiration') and (state.get('source') or {}).get('vibration',1)>0
        timeline.append({'time':t,'f0':state['f0'] if voiced else None})
    audio=np.concatenate(chunks);fade=min(480,len(audio)//4);audio[:fade]*=np.linspace(0,1,fade);audio[-fade:]*=np.linspace(1,0,fade)
    # Explicit timeline silence is an editing operation, distinct from a
    # physically occluded tract. Do not leave resonator tails in a silent frame.
    for start,end in silent_intervals:
        audio[start:end]=0
        fade_in=min(240,start);fade_out=min(240,n-end)
        if fade_in:audio[start-fade_in:start]*=np.linspace(1,0,fade_in)
        if fade_out:audio[end:end+fade_out]*=np.linspace(0,1,fade_out)
    # Apply the audition envelope after output gain/limiting, so high audition
    # gain cannot squash the attack back into an abrupt, full-volume onset.
    return {'frames':frames,'duration':duration,'audio':audio,'audition_audio':audio*np.concatenate(gains),
            'envelope':sequence_envelope(n,engine.sr,silent_intervals),'pictures':pictures,'times':times,'timeline':timeline}
