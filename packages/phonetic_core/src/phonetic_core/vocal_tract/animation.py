"""Precompute synchronized audio and geometry in the worker."""
import json
import zlib
import numpy as np
from phonetic_core.vocal_tract.trajectory import validate_frames, sample_trajectory, validate_pitch_curve
from phonetic_core.vocal_tract.envelope import sequence_envelope

def prepare_animation(engine,frames,section=65,*,pictures_enabled=True,pitch_curve=None,cancel=None):
    curve=validate_pitch_curve(pitch_curve or [])
    sample_frame=lambda frames,seconds:sample_trajectory(frames,seconds,curve)
    def glottis(pose):
        if pose.get('silent'):return engine.glottis(150,source={'mode':'voiceless','pressure_pa':0,'vibration':0})
        return engine.glottis(pose['f0'],source=pose['source']) if 'source' in pose else engine.glottis(pose['f0'])
    frames=validate_frames(engine,frames)
    # Mixed automatic/manual endpoints use their actual root coordinates.
    if any(f.get('manual_root',False) for f in frames):
        for f in frames:
            if not f['manual_root']:
                engine.set_manual_root(False)
                actual=engine.snapshot(f['params'],section,f['lip_width'])['limited']
                for key in ('TRX','TRY'):
                    i=engine.names.index(key);f['params'][i]=actual[i]
                f['manual_root']=True
    n=round(sum(f['duration'] for f in frames)*engine.sr);duration=n/engine.sr;block=960
    silent_intervals=[];elapsed=0.
    for f in frames:
        start=round(elapsed*engine.sr);elapsed+=f['duration']
        if f.get('silent'):silent_intervals.append((start,round(elapsed*engine.sr)))
    cuts=sorted({edge for interval in silent_intervals for edge in interval}|{n})
    chunks=[];gains=[];pictures=[];times=[];first=sample_frame(frames,0)
    with engine.lock:
        engine.set_manual_root(first.get('manual_root',False))
        engine.reset_tube(engine.prepare_tube(first['params'],first['lip_width']),glottis(first))
        offset=0;previous_silent=first.get('silent',False)
        while offset<n:
            if cancel is not None and cancel.is_set():raise ValueError('已取消生成')
            size=min(block,next(edge for edge in cuts if edge>offset)-offset)
            # Half-open intervals: the last sample of a voiced block belongs to
            # that block, even when its endpoint is the next silence boundary.
            pose=sample_frame(frames,(offset+size-(.25 if silent_intervals else 0))/engine.sr)
            engine.set_manual_root(pose.get('manual_root',False))
            tube=engine.prepare_tube(pose['params'],pose['lip_width'])
            if previous_silent and not pose.get('silent'):engine.reset_tube(tube,glottis(pose))
            chunks.append(engine.block_tube(tube,glottis(pose),size))
            gains.append(np.full(size,10**((pose.get('source') or {}).get('audition_gain_db',0)/20)))
            previous_silent=pose.get('silent',False);offset+=size
            # Display frames are cached before playback. Geometry computation
            # cannot delay or perturb the native audio stream during playback.
        if pictures_enabled:
            for index in range((n*30+engine.sr-1)//engine.sr+1):
                if cancel is not None and cancel.is_set():raise ValueError('已取消生成')
                time=min(index/30,duration);shown=sample_frame(frames,time)
                engine.set_manual_root(shown.get('manual_root',False))
                state=engine.snapshot(shown['params'],section,shown['lip_width']);state['f0']=shown['f0']
                state.update(source=shown.get('source'),manual_root=shown.get('manual_root',False))
                if shown.get('silent'):state['silent']=True
                pictures.append(zlib.compress(json.dumps(state,separators=(',',':')).encode(),1));times.append(time)
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
            'envelope':sequence_envelope(n,engine.sr,silent_intervals),'pictures':pictures,'times':times}
