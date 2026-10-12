"""Real native M10-R15: active resolution, single poses and cache identity."""
import copy
from pathlib import Path
import json
import zlib
import numpy as np
import pytest
from phonetic_core.vocal_tract.animation import prepare_animation
from phonetic_core.vocal_tract.trajectory import validate_frames, sample_trajectory
from ptb_desktop.vocal_tract.runtime import Runtime
from ptb_desktop.vocal_tract.video import VideoSession

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    r = Runtime(ROOT/'resources/vocal_tract/native', tmp_path_factory.mktemp('m10-r15'), playback_allowed=False)
    yield r
    r.close()


def frame(r, **kw):
    return dict(params=r.engine.presets['a'], duration=.05, f0=150, lip_width=1, **kw)


def test_default_and_resolution_roundtrip_preserve_tissue(runtime):
    e=runtime.engine
    assert e.section_count==129 and e.metadata()['section_options']==[65,129,257]
    states=[]
    for count in [129,65,257,129]:
        e.set_section_count(count);s=e.snapshot(e.presets['a'],section=999)
        assert len(s['centerline'])==len(s['airway_sections'])==count
        assert s['section']==count-1 and s['section_count']==count
        assert np.isfinite(s['tube_areas']).all()
        states.append(s)
    assert all(s['contours']==states[0]['contours'] for s in states)
    assert states[-1]['tube_areas']==states[0]['tube_areas']
    assert states[-1]['centerline']==states[0]['centerline']
    for invalid in [64,128,256,0,1000,True,129.0,'129']:
        with pytest.raises(ValueError):e.set_section_count(invalid)
    assert e.section_count==129


@pytest.mark.parametrize('silent',[False,True])
def test_single_minimum_frame_keeps_pose_and_pitch_curve(runtime,silent):
    frames=[frame(runtime,silent=silent)]
    curve=[[0,100],[1,300]]
    before=copy.deepcopy(frames)
    a=prepare_animation(runtime.engine,frames,pitch_curve=curve)
    assert a['duration']==.05 and len(a['audio'])==2400
    assert len(a['pictures'])==3 and a['times']==[0,1/30,.05]
    assert frames==before
    for time in [0,.025,.05]:
        pose=sample_trajectory(a['frames'],time,curve)
        assert pose['params']==frames[0]['params']
        assert pose['f0']==(150 if silent else 100+time/.05*200)
    if silent:assert np.count_nonzero(a['audio'])==0
    else:assert np.max(np.abs(a['audio']))>0
    with pytest.raises(ValueError,match='至少需要 1'):validate_frames(runtime.engine,[])
    assert validate_frames(runtime.engine,[],for_storage=True)==[]


def test_resolution_invalidates_cached_animation_and_rejects_stale_id(runtime):
    r=runtime;seq=[frame(r)]
    first=r.invoke('animation/prepare',{'frames':seq})
    assert r.invoke('animation/prepare',{'frames':seq})['cached']
    r.invoke('pose',dict(params=r.engine.presets['a'],revision=r.revision+1,section_count=65,section=32))
    with pytest.raises(ValueError,match='失效'):r.invoke('animation/audio',{'id':first['id']})
    low=r.invoke('animation/prepare',{'frames':seq})
    assert not low['cached'] and low['prepare_count']==first['prepare_count']+1
    assert r.invoke('animation/picture',{'id':low['id'],'index':0})['state']['section_count']==65
    r.invoke('pose',dict(params=r.engine.presets['a'],revision=r.revision+1,section_count=129,section=64))


def test_silence_moves_like_ordinary_frame_and_remains_exactly_quiet(runtime):
    e=runtime.engine
    ordinary=validate_frames(e,[dict(params=e.presets['a'],duration=.4,lip_width=.8,larynx_height=-.2,f0=120),
                               dict(params=e.presets['i'],duration=.2,lip_width=1.2,larynx_height=.2,f0=200)])
    silent=copy.deepcopy(ordinary);silent[0]['silent']=True
    for t in [0,.05,.2,.35,.399999,.4]:
        moving=sample_trajectory(silent,t,[[0,100],[1,300]])
        normal=sample_trajectory(ordinary,t)
        for key in ['params','lip_width','larynx_height','manual_root']:
            assert moving[key]==normal[key]
        if t<.4:assert moving['silent'] and moving['f0']==120
    np.testing.assert_allclose(sample_trajectory(silent,.2)['params'],(np.array(ordinary[0]['params'])+ordinary[1]['params'])/2)
    np.testing.assert_allclose(sample_trajectory(silent,.4-1e-7)['params'],ordinary[1]['params'],atol=1e-10)
    before=copy.deepcopy(silent);a=prepare_animation(e,silent)
    assert silent==before and len(a['audio'])==28800
    assert np.count_nonzero(a['audio'][:19200])==0 and np.std(a['audio'][19200:])>1e-5
    states=[json.loads(zlib.decompress(a['pictures'][i])) for i in [0,6,12,18]]
    assert states[0]['silent'] and states[1]['silent']
    assert not states[2].get('silent') and not states[3].get('silent')
    assert states[0]['contours']['tongue']!=states[1]['contours']['tongue']!=states[2]['contours']['tongue']
    np.testing.assert_allclose(states[1]['params'],sample_trajectory(silent,.2)['params'],atol=1e-9)
    assert states[2]['params']==states[3]['params']


@pytest.mark.parametrize('cancel',[False,True])
def test_mp4_failure_or_cancel_keeps_existing_target(tmp_path,monkeypatch,cancel):
    target=tmp_path/'action.mp4';target.write_bytes(b'existing user video')
    session=VideoSession(target,dict(width=1280,height=720,fps=30,duration=.05))
    # Packet content is immaterial to this failure-injection test. The actual
    # codec is separately verified with Qt WebCodecs and independent FFmpeg.
    session.opus=b'OpusHead'+bytes([1,1,0,0])+bytes(7)
    session.packets=[dict(track=1,timestamp=0,duration=33333,key=True,data=b'v'),
                     dict(track=1,timestamp=33333,duration=16667,key=False,data=b'v'),
                     dict(track=2,timestamp=0,duration=60000,key=True,data=b'a')]
    def failed(raw,temporary):
        temporary.write_bytes(b'incomplete MP4')
        if cancel:session.cancelled.set()
        else:raise ValueError('injected encoding failure')
    monkeypatch.setattr(session,'encode_mp4',failed)
    with pytest.raises(ValueError,match='取消' if cancel else 'injected'):session.finish()
    assert target.read_bytes()==b'existing user video'
    assert list(tmp_path.iterdir())==[target]
