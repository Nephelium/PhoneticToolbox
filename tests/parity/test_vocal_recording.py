"""M10 R4 real native behavior and bounded desktop file regression."""
import base64
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.signal import butter,sosfiltfilt
from phonetic_core.vocal_tract.document import make_document,sequence_document
from phonetic_core.vocal_tract.trajectory import validate_frames,sample_trajectory
from phonetic_core.vocal_tract.animation import prepare_animation
from ptb_desktop.vocal_tract.runtime import Runtime
from ptb_desktop.vocal_tract.files import VocalFiles
from ptb_desktop.vocal_tract.video import VideoSession
from ptb_desktop.vocal_tract.audio_output import audition_samples

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime(tmp_path):
    r=Runtime(ROOT/'resources/vocal_tract/native',tmp_path/'profile',playback_allowed=False)
    yield r
    r.close()


def frames(r):return [{'params':r.engine.presets[n],'name':'/'+n+'/','duration':.2} for n in ('a','i')]


def test_default_f0_and_explicit_old_pitch(runtime):
    f=validate_frames(runtime.engine,frames(runtime));assert all(p['f0']==150 for p in f)
    f[0]['f0']=125;assert validate_frames(runtime.engine,f)[0]['f0']==125
    assert runtime.live.f0==150
    assert validate_frames(runtime.engine,[{'params':runtime.engine.presets['a']}],for_storage=True)[0]['duration']==.2


def test_short_silence_has_exact_samples_no_pitch_and_portable_type(runtime):
    e=runtime.engine
    f=[{'params':e.presets['a'],'duration':.05,'f0':210},
       {'params':e.presets['a'],'duration':.05,'silent':True,'name':'停顿'},
       {'params':e.presets['i'],'duration':.05,'f0':210}]
    doc=make_document(e,f,[[0,210],[1,310]])
    assert doc['frames'][1]['silent'] is True
    assert sample_trajectory(doc['frames'],.05,doc['pitch_curve'])['silent']
    assert sample_trajectory(doc['frames'],.075,doc['pitch_curve'])['f0']==150
    assert not sample_trajectory(doc['frames'],.1,doc['pitch_curve']).get('silent',False)
    prepared=prepare_animation(e,f,pitch_curve=doc['pitch_curve'])
    assert len(prepared['audio'])==7200
    np.testing.assert_array_equal(prepared['audio'][2400:4800],np.zeros(2400))
    np.testing.assert_array_equal(prepared['audition_audio'][2400:4800],np.zeros(2400))
    assert np.std(prepared['audio'][:2400])>1e-5 and np.std(prepared['audio'][4800:])>1e-5
    first=runtime.invoke('animation/prepare',{'frames':f});assert runtime.invoke('animation/prepare',{'frames':f})['cached']
    normal=copy.deepcopy(f);normal[1].pop('silent');assert not runtime.invoke('animation/prepare',{'frames':normal})['cached']
    f[0]['duration']=.049
    with pytest.raises(ValueError,match='0.05'):validate_frames(e,f)


def test_two_minimum_silent_frames_and_invalid_type(runtime):
    f={'params':runtime.engine.presets['a'],'duration':.05,'silent':True}
    a=prepare_animation(runtime.engine,[f,f]);assert len(a['audio'])==4800 and not np.any(a['audio'])
    with pytest.raises(ValueError,match='静音帧'):validate_frames(runtime.engine,[dict(f,silent='yes'),f])


def test_fractional_duration_uses_audio_sample_clock(runtime,tmp_path):
    f={'params':runtime.engine.presets['a'],'duration':.2}
    a=prepare_animation(runtime.engine,[f]*3)
    assert a['duration']==.6 and len(a['audio'])==28800 and len(a['pictures'])==19
    assert a['times'][-1]==.6 and a['times'][-2]<.6
    session=VideoSession(tmp_path/'fraction.webm',dict(width=1280,height=720,fps=30,duration=.2+.2+.2))
    assert session.duration==.6
    assert len(validate_frames(runtime.engine,[dict(f,duration=.3)]*40))==40


def test_portable_file_roundtrip_and_invalid_file_preserve_existing(runtime,tmp_path):
    files=VocalFiles();path=tmp_path/'演示.ptb-vocal.json';body={'frames':frames(runtime),'pitch_curve':[[0,150],[1,250]]}
    assert files.invoke('document/save',body,runtime,path)['saved']
    loaded=files.invoke('document/open',{},runtime,path)['document']
    assert loaded==make_document(runtime.engine,**dict(frames=body['frames'],curve=body['pitch_curve']))
    before=path.read_bytes();broken=copy.deepcopy(loaded);broken['parameters'].reverse()
    bad=tmp_path/'bad.json';bad.write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='参数顺序'):files.invoke('document/open',{},runtime,bad)
    assert path.read_bytes()==before
    broken=copy.deepcopy(loaded);broken['frames'][0]['params'][8]=-100
    with pytest.raises(ValueError,match='超出'):sequence_document(runtime.engine,broken)
    assert files.invoke('document/save',body,runtime,None)=={'cancelled':True}


def test_presets_persist_names_and_bad_edit_does_not_overwrite(runtime):
    preset={**frames(runtime)[0],'id':'local-a','name':'自存 /a/'}
    saved=runtime.invoke('presets/save',{'presets':[preset]})
    assert runtime.invoke('presets/load')==saved
    with pytest.raises(ValueError):runtime.invoke('presets/save',{'presets':[{**preset,'name':'bad\nname'}]})
    assert runtime.invoke('presets/load')==saved


def test_cache_identical_replay_after_stop_and_each_material_edit(runtime):
    body={'frames':frames(runtime),'pitch_curve':[[0,150],[1,190]]}
    first=runtime.invoke('animation/prepare',body);runtime.invoke('animation/stop')
    cached=runtime.invoke('animation/prepare',body)
    assert cached['cached'] and cached['id']==first['id'] and cached['prepare_count']==1
    assert len(cached['times'])==13 and cached['times'][5]==5/30
    for edit in ('f0','duration','lip_width','manual_root','name','source','curve','params'):
        modified=copy.deepcopy(body)
        f=modified['frames'][0]
        if edit=='curve':modified['pitch_curve']=[[0,180],[1,190]]
        elif edit=='params':f['params'][8]+=.1
        else:f[edit]={'f0':170,'duration':.3,'lip_width':1.2,'manual_root':True,'name':'renamed','source':{'mode':'whisper'}}[edit]
        result=runtime.invoke('animation/prepare',modified)
        assert not result['cached'],edit
        assert runtime.invoke('animation/prepare',modified)['cached'],edit
    assert not runtime.live.active


def test_posterior_targets_stop_at_actual_model_wall(runtime):
    e=runtime.engine;e.set_manual_root(True);p=e.presets['a'].copy();p[8]=p[14]=-100
    s=e.snapshot(p);w=e.posterior
    back=lambda y:w['x']+(y-w['y'])*w['slope']
    assert s['params'][8]>-1 and s['params'][14]>-3
    assert all(q[0]>=back(q[1])-.00002 for q in s['contours']['tongue'][:34])
    farther=p.copy();farther[8]=farther[14]=-200
    assert e.snapshot(farther)['params']==s['params']


def test_whisper_compensation_is_audible_and_keeps_closure_quiet(runtime):
    e=runtime.engine
    def prepared(preset):
        f={'params':e.presets[preset],'duration':.5,'source':{'mode':'whisper'}}
        return prepare_animation(e,[f,f],pictures_enabled=False)
    a=prepared('a');tail=a['audition_audio'][-20000:]
    assert np.std(audition_samples(tail,.8,0))>.005
    assert np.std(tail)/np.std(a['audio'][-20000:])==pytest.approx(10**(24/20))
    t=prepared('t')
    # Closed tract has a sub-audible pressure/DC relaxation, plus the release
    # envelope. Check the audible band, not DC drift interpreted as airflow noise.
    audible=sosfiltfilt(butter(2,20,fs=e.sr,btype='highpass',output='sos'),audition_samples(t['audition_audio'],.8,0))
    assert np.std(audible[-20000:])<1e-6


def test_export_audio_uses_same_prepared_samples_and_gain(runtime):
    prepared=runtime.invoke('animation/prepare',{'frames':frames(runtime)})
    encoded=runtime.invoke('animation/audio',{'id':prepared['id']})
    actual=np.frombuffer(base64.b64decode(encoded['base64']),dtype='<f4')
    expected=audition_samples(runtime.animation['audition_audio'],.8,0).astype('<f4')
    np.testing.assert_array_equal(actual,expected)
    assert len(actual)==round(prepared['duration']*48000)


def test_incomplete_video_and_cancel_preserve_target(tmp_path):
    target=tmp_path/'video.webm';target.write_bytes(b'original')
    session=VideoSession(target,dict(width=1280,height=720,fps=30,duration=1))
    with pytest.raises(ValueError,match='不完整'):session.finish()
    session.cancel();assert target.read_bytes()==b'original'
    with pytest.raises(ValueError,match='结束'):session.append({'packets':[]})
