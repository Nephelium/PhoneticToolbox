"""R3 recording selection, independent PTS/PCM readback and single offset application."""
import json
from pathlib import Path
import sys
import wave
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/p) for p in ('backend/src','desktop/src','packages/phonetic_core/src')]

@pytest.fixture
def media(tmp_path):
    from ptb_desktop.m05_capture import CaptureMux
    source=tmp_path/'source.mkv';mux=CaptureMux(source,64,64,30,48000)
    audio=np.zeros((24000,1),np.int16);audio[9600:9700]=8000
    mux.audio_frame(audio,50_000_000)
    for i,t in enumerate((.1,.14,.23,.31,.47)):mux.video_frame(np.full((64,64,3),i*40,np.uint8),round(t*1e9))
    mux.close()
    points=[[32+15*np.cos(i*2*np.pi/478),32+20*np.sin(i*2*np.pi/478)] for i in range(478)]
    capture=dict(backend='synthetic-candidate',settings=dict(mode='realtime'),clock_mapping=dict(candidate_time_base='recording_start_estimate'),lip_manual_offset=.123,
                 frames=[dict(index=i,time_s=t,detected=True,points=points,metrics={k:float(i+1) for k in ('area','outer_width','open','circularity')},input_resolution=[64,64]) for i,t in enumerate((.1,.23,.47))])
    return source,capture,audio

@pytest.mark.parametrize('video,animation',[(False,False),(False,True),(True,False),(True,True)])
def test_save_matrix_and_clock(media,tmp_path,video,animation):
    import av
    from ptb_worker.m05_media_export import recording_bundle
    from ptb_worker.io.lip import decode_lip
    source,capture,audio=media;capture['save_options']=dict(video=video,animation=animation)
    meta=tmp_path/'capture.json';meta.write_text(json.dumps(capture),'utf8');target=tmp_path/'saved';target.mkdir()
    info=recording_bundle(source,target,metadata_path=meta)
    assert (target/'raw_recording.mp4').exists()==video
    assert (target/'face_animation.mp4').exists()==animation
    assert not any(p.name.startswith('.') for p in target.iterdir())
    values=decode_lip((target/'audio_recording.lip.json').read_bytes())
    np.testing.assert_allclose(values['relative_times'],[.05,.18,.42],rtol=0,atol=1e-15)
    assert values['metadata']['lip_manual_offset']==.123
    with wave.open(str(target/'audio_recording.wav'),'rb') as w:np.testing.assert_array_equal(np.frombuffer(w.readframes(w.getnframes()),'<i2'),audio.ravel())
    if video:
        with av.open(str(target/'raw_recording.mp4')) as c:np.testing.assert_allclose([float(f.pts*f.time_base) for f in c.decode(video=0)],[.05,.09,.18,.26,.42],atol=1/90000)
    if animation:
        with av.open(str(target/'face_animation.mp4')) as c:assert c.streams.audio and c.streams.video
        assert info['animation']['lip_manual_offset']==.123
    assert set(p.name for p in target.iterdir())=={f['name'] for f in info['files']}|{'recording-export.json'}

def test_high_rate_media_only_and_waveform_anchor(media,tmp_path):
    from ptb_worker.m05_media_export import recording_bundle,inspect_recording
    source,capture,audio=media;capture.update(settings=dict(mode='record_then_analyze'),frames=[],save_options=dict(video=True,animation=False))
    meta=tmp_path/'capture.json';meta.write_text(json.dumps(capture),'utf8');target=tmp_path/'saved';target.mkdir()
    recording_bundle(source,target,metadata_path=meta)
    assert set(p.name for p in target.iterdir())=={'raw_recording.mp4','audio_recording.wav','recording-export.json'}
    scratch=tmp_path/'inspection';scratch.mkdir();info=inspect_recording(source,scratch)
    assert info['audio']['first_decoded_pts_s']==.05 and info['first_video_pts_s']==.1
    wave=info['waveform'];peak=max(range(len(wave['values'])),key=wave['values'].__getitem__)
    assert wave['times'][peak]==.2 and wave['values'][peak]==8000/32768

@pytest.mark.parametrize('save_video',[False,True])
def test_duplicate_video_pts_keeps_original_and_does_not_block_audio_lips(tmp_path,save_video):
    import av
    from fractions import Fraction
    from ptb_worker.m05_media_export import recording_bundle
    source=tmp_path/'duplicate.mkv'
    with av.open(str(source),'w') as out:
        v=out.add_stream('ffv1',rate=30);v.width=v.height=64;v.pix_fmt='yuv420p';v.time_base=v.codec_context.time_base=Fraction(1,1000)
        a=out.add_stream('pcm_s16le',rate=48000);a.layout='mono'
        af=av.AudioFrame.from_ndarray(np.zeros((1,24000),np.int16),format='s16',layout='mono');af.sample_rate=48000;af.time_base=Fraction(1,48000);af.pts=0
        for packet in a.encode(af):out.mux(packet)
        for i,pts in enumerate((0,33,33,100)):
            frame=av.VideoFrame.from_ndarray(np.full((64,64,3),i*50,np.uint8),format='rgb24');frame.pts=pts;frame.time_base=Fraction(1,1000)
            for packet in v.encode(frame):out.mux(packet)
        for stream in (v,a):
            for packet in stream.encode():out.mux(packet)
    capture=dict(settings=dict(mode='realtime'),save_options=dict(video=save_video,animation=False),clock_mapping=dict(candidate_time_base='recording_start_estimate'),frames=[dict(time_s=t,metrics=dict(open=1)) for t in (.01,.1)])
    metadata=tmp_path/'capture.json';metadata.write_text(json.dumps(capture),'utf8');target=tmp_path/'saved';target.mkdir()
    info=recording_bundle(source,target,metadata_path=metadata)
    assert not info['video_pts_monotonic']
    assert (target/'audio_recording.wav').is_file() and (target/'audio_recording.lip.json').is_file()
    assert not (target/'raw_recording.mp4').exists()
    if save_video:
        assert info['video_file']=='raw_recording.mkv'
        assert (target/'raw_recording.mkv').read_bytes()==source.read_bytes()
        assert info['video_timing_warning']
