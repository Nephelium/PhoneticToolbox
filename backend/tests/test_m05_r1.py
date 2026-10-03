"""M05-R1: full playback, real codecs, and saved companion offset/readers."""
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit, parse_qs
from uuid import uuid4
import wave
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'backend/src'),str(ROOT/'desktop/src'),str(ROOT/'packages/phonetic_core/src')]


def test_audio_pts_difference_has_two_rounding_endpoints(tmp_path):
    import av
    from fractions import Fraction
    from ptb_worker.m05_audio import extract_audio
    source=tmp_path/'rounded.mkv'
    with av.open(str(source),'w') as out:
        stream=out.add_stream('pcm_s16le',rate=48000);stream.layout='mono'
        for t in (.129,.190,.249,.310):
            frame=av.AudioFrame.from_ndarray(np.zeros((1,2880),np.int16),format='s16',layout='mono')
            frame.sample_rate=48000;frame.time_base=Fraction(1,48000);frame.pts=round(t*48000)
            for packet in stream.encode(frame):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
    info,_=extract_audio(source,tmp_path)
    assert info['samples']==4*2880 and info['inserted_silence_samples']==0


def test_full_replay_reads_original_jsonl_with_bounded_pages_and_seek():
    from ptb_desktop.m05_replay import ReplayReader
    frames=[dict(index=i,time_s=i/30,detected=i%7!=0,points=[[i,1]]*478,metrics=dict(open=i)) for i in range(1301)]
    data=b''.join(json.dumps(f).encode()+b'\n' for f in frames)
    calls=[]
    class Service:
        def binary(self,url):
            q=parse_qs(urlsplit(url).query);start,size=int(q['offset'][0]),int(q['size'][0]);calls.append(size)
            return data[start:start+size]
    job=dict(state='succeeded',operation='lip_analysis',result_manifest=dict(files=[dict(id='a',name='frames.jsonl',size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest())]))
    reader=ReplayReader()
    for start in (0,90,1260,360,90):
        value=reader.read(Service(),job,start)
        assert value['rows']==frames[start:start+90]
        assert value['complete']==(start+90>=len(frames))
    assert max(calls)<=1048576
    with pytest.raises(Exception):reader.read(Service(),job,1)


def test_recording_audio_preserves_samples_despite_packet_clock_jitter(tmp_path):
    import av
    from fractions import Fraction
    from ptb_worker.m05_audio import extract_audio
    source=tmp_path/'jitter.mkv'
    samples=np.arange(4*2880,dtype=np.int16)
    with av.open(str(source),'w') as out:
        stream=out.add_stream('pcm_s16le',rate=48000);stream.layout='mono'
        for i,t in enumerate((.033,.091,.155,.213)):
            frame=av.AudioFrame.from_ndarray(samples[i*2880:(i+1)*2880].reshape(1,-1),format='s16',layout='mono')
            frame.sample_rate=48000;frame.time_base=Fraction(1,48000);frame.pts=round(t*48000)
            for packet in stream.encode(frame):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
    continuous=tmp_path/'continuous';continuous.mkdir()
    info,_=extract_audio(source,continuous,clock_policy='decoded_samples')
    assert info['samples']==len(samples) and info['inserted_silence_samples']==0
    assert info['max_timestamp_residual_samples']==96
    with wave.open(str(continuous/'audio_recording.wav'),'rb') as w:
        np.testing.assert_array_equal(np.frombuffer(w.readframes(w.getnframes()),'<i2'),samples)
    strict=tmp_path/'strict';strict.mkdir()
    with pytest.raises(ValueError,match='m05_audio_overlapping_samples'):
        extract_audio(source,strict)


def test_recording_bundle_vfr_audio_and_candidate_exchange(tmp_path):
    import av
    from ptb_desktop.m05_capture import CaptureMux
    from ptb_worker.m05_media_export import recording_bundle
    from ptb_worker.io.lip import decode_lip
    source=tmp_path/'source.mkv';mux=CaptureMux(source,64,64,30,48000)
    video_times=[.1,.14,.23,.31,.47]
    samples=(np.sin(np.arange(24576)*2*np.pi*440/48000)*2000).astype(np.int16).reshape(-1,1)
    for i in range(24):
        mux.audio_frame(samples[i*1024:(i+1)*1024],round(i*1024/48000*1e9))
        if i in (4,6,10,14,22):mux.video_frame(np.zeros((64,64,3),np.uint8),round(video_times[(4,6,10,14,22).index(i)]*1e9))
    mux.close();target=tmp_path/'saved';target.mkdir()
    capture=dict(backend='candidate-test',frames=[dict(time_s=t,metrics={k:i+.25 for k in ('area','outer_width','open','circularity')}) for i,t in enumerate((0.,.09,.14))])
    (target/'capture.m05-preview.json').write_text(json.dumps(capture),'utf8')
    info=recording_bundle(source,target)
    assert info['video_frames']==5 and info['audio']['present']
    with av.open(str(target/'raw_recording.mp4')) as c:
        assert [round(float(f.pts*f.time_base),5) for f in c.decode(video=0)]==video_times
    with wave.open(str(target/'audio_recording.wav'),'rb') as w:
        assert w.getframerate()==48000
        np.testing.assert_array_equal(np.frombuffer(w.readframes(w.getnframes()),dtype='<i2'),samples.ravel())
    data=decode_lip((target/'candidate.lip.json').read_bytes())
    assert data['relative_times']==[.1,.19,.24000000000000002]
    assert info['candidate_not_legacy'] and info['candidate_frames']==3
    companion=target/'audio_recording.lip.json'
    assert companion.read_bytes()==(target/'candidate.lip.json').read_bytes()
    assert decode_lip(companion.read_bytes())['metadata']['source_backend']=='candidate-test'
    assert decode_lip(companion.read_bytes())['metadata']['source_status']=='candidate'
    assert info['audio']['signal']['peak_dbfs'] > -30
    assert not info['audio']['signal']['low_signal']
    assert all(entry['sha256']==hashlib.sha256((target/entry['name']).read_bytes()).hexdigest() for entry in info['files'])


def test_saved_offset_is_on_the_auto_associated_file_and_original_is_kept(tmp_path):
    from ptb_desktop.m05_bridge import M05Bridge
    from ptb_desktop.file_provider import FileProvider
    from ptb_worker.io.lip import encode_lip,decode_lip,convert_local_legacy_lip
    from ptb_worker.io.annotation import lip_preview
    import importlib.util
    spec=importlib.util.spec_from_file_location('m05_r1_acoustic_lip',ROOT/'packages/phonetic_core/src/phonetic_core/acoustic/lip.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    resolve_lip_time_axis=module.resolve_lip_time_axis
    from types import SimpleNamespace
    import pickle
    data=dict(relative_times=[0.,.1,.2],area=[1.,2.,3.],outer_width=[2.,3.,4.],open=[3.,4.,5.],circularity=[.2,.3,.4],metadata=dict(time_alignment_mode='anchored_audio_start',audio_first_frame_time=0.,lip_manual_offset=0.))
    payload=encode_lip(data);file=dict(id='exchange',name='audio_recording.lip.json',sha256=hashlib.sha256(payload).hexdigest(),size_bytes=len(payload))
    class Service:
        def get(self,url):return dict(id=str(uuid4()),state='succeeded',operation='lip_analysis',result_manifest=dict(files=[file]))
        def binary(self,url):
            q=parse_qs(urlsplit(url).query);return payload[int(q['offset'][0]):int(q['offset'][0])+int(q['size'][0])]
    provider=FileProvider();grant=provider.choose('output',lambda:tmp_path)
    bridge=M05Bridge(SimpleNamespace(service=Service(),provider=provider))
    bridge.invoke(dict(op='m05_save',job=str(uuid4()),directory=grant['id'],offset=.125,action='apply'))
    saved=next(tmp_path.iterdir());current=(saved/'audio_recording.lip.json').read_bytes()
    decoded=decode_lip(current)
    assert decoded['metadata']['lip_manual_offset']==.125
    assert (saved/'unaligned.lip.json').read_bytes()==payload
    axis=resolve_lip_time_axis(decoded)
    np.testing.assert_array_equal(axis.times+axis.manual_offset,np.array([.125,.225,.325]))
    assert set(module.interpolate_lip(decoded,axis.times+axis.manual_offset))=={'LipArea','LipWidth','LipOpen','LipCirc'}
    assert lip_preview(current,'audio_recording.lip.json')['data']['metadata']['lip_manual_offset']==.125
    # Old V2 numeric PKL and its companion enter the same inert representation.
    legacy={**data,'landmarks':np.zeros((3,478,2),np.float32),'area':np.array(data['area'])}
    restored=decode_lip(convert_local_legacy_lip(pickle.dumps(legacy)))
    np.testing.assert_array_equal(restored['area'],data['area'])
