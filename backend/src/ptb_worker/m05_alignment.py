"""Bounded invocation of the unchanged V2 offset estimator; never auto-applied."""
import wave
import numpy as np
from .m05_results import rows
from phonetic_core.lip.alignment import estimate_lip_audio_offset_seconds

def suggest_offset(directory,metadata):
    if not metadata['audio']['present']:return dict(available=False,reason='no_audio')
    with wave.open(str(directory/'audio_recording.wav'),'rb') as audio:
        if audio.getnframes()>6_000_000:return dict(available=False,reason='suggestion_sample_budget',limit_samples=6_000_000)
        if audio.getnchannels()!=1:return dict(available=False,reason='legacy_estimator_requires_mono',channels=audio.getnchannels())
        samples=np.frombuffer(audio.readframes(audio.getnframes()),dtype='<i2').astype(np.float32)/32768
        rate=audio.getframerate()
    times=[];opening=[]
    for row in rows(directory/'frames.jsonl'):
        times.append(row['time_s']);value=(row['metrics'] or {}).get('open');opening.append(np.nan if value is None else value)
    offset=estimate_lip_audio_offset_seconds(samples,rate,np.asarray(times),np.asarray(opening))
    return dict(available=True,offset_seconds=float(offset),method='V2 estimate_lip_audio_offset_seconds unchanged',applied=False,
                warning='Correlation suggestion only; not acoustic or physiological synchronization validation',audio_samples=len(samples),lip_frames=len(times))
