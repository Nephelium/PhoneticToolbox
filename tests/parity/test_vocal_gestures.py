"""M10: native independent lips, explicit release/VOT and overlapping regions."""
import copy
import json
import os
from pathlib import Path
import numpy as np
import pytest
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.trajectory import validate_frames, sample_trajectory
from phonetic_core.vocal_tract.animation import prepare_animation
from phonetic_core.vocal_tract.document import make_document, sequence_document
from phonetic_core.vocal_tract.preset_pack import make_pack, validate_pack
from phonetic_core.vocal_tract.suprasegmentals import validate_features, compile_features, apply_features
from phonetic_core.vocal_tract.source import validate_source

ROOT=Path(__file__).resolve().parents[2]
@pytest.fixture
def engine():
    e=Engine(resource_dir=Path(os.environ.get('PTB_M10_TEST_RESOURCES',ROOT/'resources/vocal_tract/native')))
    yield e
    e.close()

def pose(e,name='a',**kw):
    return dict(params=e.presets[name],lip_width=1.,f0=120.,duration=.2,name=name,source=validate_source(None),**kw)


def test_native_lips_move_independently_and_change_acoustic_area(engine):
    baseline=engine.snapshot(engine.params)
    for coordinate in range(4):
        lip=[0.]*4;lip[coordinate]=.3
        state=engine.snapshot(engine.params,lip_shape=lip)
        moving='upper_lip' if coordinate<2 else 'lower_lip';fixed='lower_lip' if coordinate<2 else 'upper_lip'
        assert state['contours'][fixed]==baseline['contours'][fixed]
        assert np.max(np.abs(np.array(state['contours'][moving])-baseline['contours'][moving]))>.29
        assert np.max(np.abs(np.array(state['tube_areas'])-baseline['tube_areas']))>.05
        assert np.isfinite(state['tube_areas']).all()
    zero=engine.snapshot(engine.params,lip_shape=[0.]*4)
    assert zero['contours']==baseline['contours'] and zero['tube_areas']==baseline['tube_areas']


def test_explicit_closure_release_and_signed_vot_are_separate(engine):
    stop=pose(engine,'t');stop.update(source=validate_source({'mode':'voiceless'}),stop_timing=dict(closure=.06,release=.008,aspiration=.132,vot=.14))
    frames=validate_frames(engine,[stop,pose(engine)])
    assert sample_trajectory(frames,.059)['params']==frames[0]['params']
    np.testing.assert_allclose(sample_trajectory(frames,.069)['params'],frames[1]['params'],atol=1e-12)
    assert sample_trajectory(frames,.199)['source']['vibration']==0
    assert sample_trajectory(frames,.203)['source']['vibration']==1
    negative=copy.deepcopy(frames);negative[0]['stop_timing']['vot']=-.03
    assert sample_trajectory(negative,.02)['source']['vibration']==0
    assert sample_trajectory(negative,.035)['source']['vibration']==1
    assert sample_trajectory(negative,.035)['params']==frames[0]['params']
    rendered=prepare_animation(engine,frames,pictures_enabled=False)
    y=rendered['audio'];rms=lambda a,b:np.sqrt(np.mean(y[round(a*engine.sr):round(b*engine.sr)]**2))
    assert rms(.06,.08)>2*rms(.03,.055), (rms(.06,.08),rms(.03,.055))
    assert np.isfinite(y).all()
    with pytest.raises(ValueError,match='后续释放目标'):validate_frames(engine,[stop])


def test_overlapping_regions_sum_offsets_after_last_fixed_and_follow_frame_ids(engine):
    frames=[pose(engine,frame_id='a'),pose(engine,'i',frame_id='i')]
    def feature(key,mode,value,selection):return dict(id=key,parameter='larynx_height',mode=mode,value=value,enabled=True,selection=selection)
    by_id={'kind':'frames','ids':['a']};by_time={'kind':'time','start':.1,'end':.3}
    features=validate_features(engine,[feature('first','fixed',-.3,by_id),feature('offset','offset',.4,by_id),feature('last','fixed',.8,by_time)])
    baseline=pose(engine);original=copy.deepcopy(baseline);compiled=compile_features(features,frames)
    assert apply_features(engine,baseline,compiled,.05,.4)['larynx_height']==pytest.approx(.1)
    assert apply_features(engine,baseline,compiled,.15,.4)['larynx_height']==1.
    assert apply_features(engine,baseline,compiled,.25,.4)['larynx_height']==.8
    assert apply_features(engine,baseline,compiled,.35,.4)==baseline
    reordered=compile_features(features,list(reversed(frames)))
    assert apply_features(engine,baseline,reordered,.05,.4)==baseline
    assert apply_features(engine,baseline,reordered,.35,.4)['larynx_height']==pytest.approx(.1)
    assert baseline==original


def test_document_and_pack_preserve_new_fields_and_old_documents(engine):
    frame=pose(engine,frame_id='a',lip_shape=[0.,.1,-.2,.3]);features=[dict(id='lip',parameter='lower_lip_y',mode='offset',value=.1,enabled=True,selection={'kind':'frames','ids':['a']})]
    document=make_document(engine,[frame],[],features=features)
    assert document['version']==8 and document['features']==features
    restored=sequence_document(engine,json.loads(json.dumps(document)))
    assert restored['frames'][0]['lip_shape']==frame['lip_shape'] and restored['frames'][0]['frame_id']=='a'
    pack=make_pack(engine,{'presets':[frame]});assert pack['version']==3
    assert 'frame_id' not in pack['presets'][0] and validate_pack(engine,pack)['presets'][0]['lip_shape']==frame['lip_shape']
    legacy={**document,'version':7,'geometry_method':'m10/r19','model':'VTL-2.4-JD2-M10-R19','frames':[pose(engine)],'features':[]}
    assert 'geometry_notice' in sequence_document(engine,legacy)
    legacy['frames']=[frame]
    with pytest.raises(ValueError,match='版本 8'):sequence_document(engine,legacy)


def test_frame_identity_includes_aspiration_and_rejects_duplicates(engine):
    values=[pose(engine,frame_id='a'),dict(aspiration=True,duration=.05,frame_id='h')]
    assert validate_frames(engine,values,for_storage=True)[1]['frame_id']=='h'
    values[1]['frame_id']='a'
    with pytest.raises(ValueError,match='重复'):validate_frames(engine,values,for_storage=True)


@pytest.mark.parametrize('speaker,up',[('JD2',.6),('W02',.8)])
def test_lower_lip_constriction_has_native_frication_and_model_specific_offsets(speaker,up):
    engine=Engine(speaker,resource_dir=Path(os.environ.get('PTB_M10_TEST_RESOURCES',ROOT/'resources/vocal_tract/native')))
    try:
        lip=[0.,0.,-.6,up];state=engine.snapshot(engine.presets['a'],lip_shape=lip)
        areas=np.asarray(state['tube_areas'])
        assert areas.argmin()>=len(areas)-4 and 0<areas.min()<.2
        frame=pose(engine,lip_shape=lip);frame['source']=validate_source({'mode':'voiceless'})
        audio=prepare_animation(engine,[frame],pictures_enabled=False)['audio'][2400:7200]
        power=np.abs(np.fft.rfft(audio))**2;hz=np.fft.rfftfreq(len(audio),1/engine.sr)
        assert np.sqrt(np.mean(audio**2))>1e-4
        assert power[hz>1000].sum()/power.sum()>.85
    finally:engine.close()


def test_completed_animation_retains_independent_lips_for_next_live_sound(engine):
    from phonetic_core.vocal_tract.session import Runtime
    from ptb_worker.vocal_web import BrowserAudio,MemoryProfile
    live=BrowserAudio(engine);runtime=Runtime(engine,MemoryProfile(),live)
    frame=pose(engine,lip_shape=[0.,.1,-.2,.3]);clip=runtime.invoke('animation/prepare',{'frames':[frame]})
    runtime.invoke('animation/play',{'id':clip['id']})
    live.pull({'samples':9600});live.acknowledge({'position':.2,'completed':True})
    result=runtime.invoke('animation/frame',{'id':clip['id']})
    assert result['completed'] and live.lip_shape==frame['lip_shape']
