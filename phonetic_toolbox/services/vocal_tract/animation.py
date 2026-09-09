"""Precompute synchronized audio and geometry in the worker."""
import json
import zlib
import numpy as np
from phonetic_toolbox.core.vocal_tract.trajectory import validate_frames, sample_trajectory, validate_pitch_curve

def prepare_animation(engine,frames,section=65,*,pictures_enabled=True,pitch_curve=None):
    curve=validate_pitch_curve(pitch_curve or [])
    sample_frame=lambda frames,seconds:sample_trajectory(frames,seconds,curve)
    glottis=lambda pose:engine.glottis(pose['f0'],source=pose['source']) if 'source' in pose else engine.glottis(pose['f0'])
    frames=validate_frames(engine,frames);duration=sum(f['duration'] for f in frames);n=round(duration*engine.sr);block=960
    chunks=[];pictures=[];times=[];first=sample_frame(frames,0)
    with engine.lock:
        engine.reset_tube(engine.prepare_tube(first['params'],first['lip_width']),glottis(first))
        for offset in range(0,n,block):
            size=min(block,n-offset);pose=sample_frame(frames,(offset+size)/engine.sr)
            tube=engine.prepare_tube(pose['params'],pose['lip_width'])
            chunks.append(engine.block_tube(tube,glottis(pose),size))
            # Display frames are cached before playback. Geometry computation
            # cannot delay or perturb the native audio stream during playback.
            if pictures_enabled and offset%(block*3)==0:
                time=offset/engine.sr;shown=sample_frame(frames,time)
                state=engine.snapshot(shown['params'],section,shown['lip_width']);state['f0']=shown['f0']
                state['source']=shown.get('source')
                pictures.append(zlib.compress(json.dumps(state,separators=(',',':')).encode(),1));times.append(time)
        if pictures_enabled:
            final=sample_frame(frames,duration);state=engine.snapshot(final['params'],section,final['lip_width']);state['f0']=final['f0']
            state['source']=final.get('source')
            pictures.append(zlib.compress(json.dumps(state,separators=(',',':')).encode(),1));times.append(duration)
    audio=np.concatenate(chunks);fade=min(480,len(audio)//4);audio[:fade]*=np.linspace(0,1,fade);audio[-fade:]*=np.linspace(1,0,fade)
    return {'frames':frames,'duration':duration,'audio':audio,'pictures':pictures,'times':times}
