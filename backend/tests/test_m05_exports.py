"""Actual codec round trips, synthetic clock offsets, no physical devices."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'backend/src'),str(ROOT/'desktop/src'),str(ROOT/'packages/phonetic_core/src')]

def test_driver_zero_adc_is_unavailable_not_repeated_zero_pts():
    from ptb_desktop.m05_capture import audio_clock_observation
    stamp,valid=audio_clock_observation(1_955_361_600,0.,948085.496335,1920,48000)
    assert valid is False and stamp==1_915_361_600
    stamp,valid=audio_clock_observation(1_955_361_600,948085.456335,948085.496335,1920,48000)
    assert valid is True and abs(stamp-1_915_361_600)<=1

def test_offset_suggestion_reuses_independent_v2_oracle(tmp_path):
    import wave
    from phonetic_core.lip.alignment import estimate_lip_audio_offset_seconds
    from ptb_worker.m05_alignment import suggest_offset
    d=json.loads((ROOT/'tests/fixtures/m05/offset-v2.json').read_text('utf8'))
    assert estimate_lip_audio_offset_seconds(np.array(d['audio']),d['sample_rate'],np.array(d['lip_times']),np.array(d['lip_open']))==d['expected']
    assert suggest_offset(tmp_path,dict(audio=dict(present=False)))['reason']=='no_audio'
    with wave.open(str(tmp_path/'audio_recording.wav'),'wb') as output:
        output.setparams((2,2,48000,0,'NONE','not compressed'));output.writeframes(bytes(16))
    assert suggest_offset(tmp_path,dict(audio=dict(present=True)))['reason']=='legacy_estimator_requires_mono'

def test_native_mux_preserves_vfr_and_audio_clock(tmp_path):
    import av
    from ptb_desktop.m05_capture import CaptureMux
    from ptb_worker.m05_audio import extract_audio
    from ptb_worker.m05_video import analyze_video,JsonLinesSink
    from phonetic_core.lip.sequence import LipConfig
    import io
    source=tmp_path/'synthetic.mkv';mux=CaptureMux(source,64,64,30,48000)
    times=[200_000_000,240_000_000,320_000_000,440_000_000]
    image=np.zeros((64,64,3),np.uint8)
    for i in range(24):
        mux.audio_frame(np.zeros((1024,1),np.int16),round(i*1024/48000*1e9))
        if i in (9,11,15,21):mux.video_frame(image,times[(9,11,15,21).index(i)])
    mux.close()
    with av.open(str(source)) as container:assert [round(float(f.pts*f.time_base),3) for f in container.decode(video=0)]==[.2,.24,.32,.44]
    audio,names=extract_audio(source,tmp_path);assert audio['samples']==24576 and audio['first_decoded_pts_s']==0 and audio['inserted_silence_samples']<=1
    sink=io.BytesIO();metadata=analyze_video(source,JsonLinesSink(sink,1_000_000),LipConfig(False))
    assert metadata['timing']['anchor_s']==0
    assert [r['time_s'] for r in map(json.loads,sink.getvalue().splitlines())]==[.2,.24,.32,.44]
    assert metadata['validity']['missing']==4

@pytest.mark.parametrize('format',['mp4','gif'])
@pytest.mark.parametrize('quality',['high','standard','small'])
def test_streaming_animation_quality_and_offset(tmp_path,format,quality):
    import av
    from ptb_worker.m05_animation import export_animation
    from ptb_worker.m05_results import export_tables
    from ptb_worker.m05_video import analyze_video,JsonLinesSink
    from phonetic_core.lip.sequence import LipConfig
    source=ROOT/'output/validation/m05/inputs/front/input.mkv';frames=tmp_path/'frames.jsonl'
    with frames.open('xb') as stream:meta=analyze_video(source,JsonLinesSink(stream,5_000_000),LipConfig(False))
    export_tables(frames,tmp_path,meta)
    target=tmp_path/('animation.'+format)
    info=export_animation(frames,target,meta,quality=quality,format=format,offset=.125)
    with av.open(str(target)) as container:
        decoded=list(container.decode(video=0))
    assert len(decoded)==info['frames']
    assert max(decoded[0].width,decoded[0].height)=={'high':1080,'standard':720,'small':540}[quality]
    # Positive offset produces a genuinely blank initial visualization, not a
    # changed measurement or an offset merely written in metadata.
    assert np.min(decoded[0].to_ndarray(format='rgb24'))>=250
    assert np.min(decoded[10].to_ndarray(format='rgb24'))<100
    with pytest.raises(FileExistsError):export_animation(frames,target,meta)

def test_compatibility_fill_and_safe_exchange_are_distinct(tmp_path):
    from ptb_worker.m05_results import export_tables
    from ptb_worker.io.lip import decode_lip
    import csv
    rows=[dict(index=i,time_s=t,detected=i==1,points=None,metrics={'open':.2} if i==1 else None) for i,t in enumerate((0.,.1,.25))]
    source=tmp_path/'frames.jsonl';source.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    export_tables(source,tmp_path,dict(timing=dict(decoded_frames=3),backend='test'))
    with (tmp_path/'measurements.csv').open(encoding='utf-8-sig') as stream:observed=list(csv.DictReader(stream))
    with (tmp_path/'legacy-compatibility.csv').open(encoding='utf-8-sig') as stream:held=list(csv.DictReader(stream))
    assert [r['open'] for r in observed]==['','0.2','']
    assert [r['imputed'] for r in held]==['True','False','True']
    data=decode_lip((tmp_path/'audio_recording.lip.json').read_bytes());assert np.isnan(data['open'][0]) and data['open'][1]==.2

def test_mp4_audio_mux_and_offset_are_real(tmp_path):
    import av
    import wave
    from ptb_worker.m05_animation import export_animation
    frames=tmp_path/'frames.jsonl'
    frames.write_text('\n'.join(json.dumps(dict(index=i,time_s=i*.1,detected=False,points=None,metrics=None)) for i in range(11))+'\n')
    audio=tmp_path/'audio.wav';samples=(np.sin(np.arange(48000)*2*np.pi*440/48000)*1000).astype('<i2')
    with wave.open(str(audio),'wb') as output:output.setparams((1,2,48000,0,'NONE','not compressed'));output.writeframes(samples.tobytes())
    meta=dict(coordinates=dict(resolutions=[[64,64]]),timing=dict(last_video_pts_s=1.,anchor_s=0.),audio=dict(first_decoded_pts_s=0.))
    result=export_animation(frames,tmp_path/'out.mp4',meta,quality='small',offset=-.125,audio_path=audio)
    assert result['audio_included'] and result['output_origin_s']==-.125
    with av.open(str(tmp_path/'out.mp4')) as container:
        decoded=list(container.decode(audio=0));first=float(decoded[0].pts*decoded[0].time_base)
    # AAC packet priming is one codec frame (1024/48000 seconds), documented
    # separately from source PCM time. Original sample records never move.
    from fractions import Fraction
    assert decoded[0].pts*decoded[0].time_base+Fraction(1024,48000)==Fraction(1,8)
    assert sum(f.samples for f in decoded)>=48000
