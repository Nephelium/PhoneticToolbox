"""Reported tongue shoulder and real regional jitter/shimmer regression."""
import copy
import json
import os
import threading
import zlib
from pathlib import Path
import numpy as np
import pytest
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.animation import prepare_animation
from phonetic_core.vocal_tract.sequence_controls import validate_global_controls, apply_global_controls
from phonetic_core.vocal_tract.suprasegmentals import validate_features, compile_features, apply_features
from phonetic_core.vocal_tract.document import make_document, sequence_document

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture
def engine():
    e=Engine(resource_dir=Path(os.environ.get('PTB_M10_TEST_RESOURCES',ROOT/'resources/vocal_tract/native')))
    yield e
    e.close()

def frame(e):
    return dict(params=list(e.presets['a']),lip_width=1.,f0=120.,duration=.3,name='a',frame_id='a')

def feature(parameter,value,mode='fixed'):
    return dict(id=parameter,enabled=True,parameter=parameter,mode=mode,value=value,
                selection=dict(kind='time',start=.1,end=.2))

def test_manual_blade_depression_has_no_acute_shoulder(engine):
    # Public author preset + reproducible manual blade/surface edits. R20 had
    # a 102.47 degree turn (77.53 degree interior angle) at rib 16.
    p=json.loads((ROOT/'frontend/public/vocal-tract/default-presets.json').read_text('utf-8'))['presets']['l']['params']
    p[13]-=.8
    state=engine.snapshot(p,tongue_shape=[.5,0,0,0])
    contour=np.asarray(state['contours']['tongue'])[:34,:2]
    d=np.diff(contour,axis=0)
    turns=np.degrees(np.arccos(np.clip(np.sum(d[:-1]*d[1:],axis=1)/
        (np.linalg.norm(d[:-1],axis=1)*np.linalg.norm(d[1:],axis=1)),-1,1)))
    assert turns.max()<90
    assert np.isfinite(state['tube_areas']).all()
    assert np.min(state['tube_areas'])>=0
    tube=engine.prepare_tube(p,tongue_shape=[.5,0,0,0])
    np.testing.assert_allclose(tube['areas'],state['tube_areas'],atol=5e-7,rtol=0)

def test_source_controls_compose_clamp_and_preserve_seed(engine):
    pose=frame(engine)|{'perturbation':{'jitter':7.,'shimmer':12.,'seed':42}}
    original=copy.deepcopy(pose)
    controls=validate_global_controls(engine,{'jitter':{'mode':'offset','value':5},'shimmer':{'mode':'fixed','value':95}})
    global_pose=apply_global_controls(engine,pose,controls)
    features=validate_features(engine,[feature('jitter',-20,'offset'),feature('shimmer',10,'offset')])
    compiled=compile_features(features,[pose])
    assert apply_features(engine,global_pose,compiled,.099,.3)['perturbation']=={'jitter':12.,'shimmer':95.,'seed':42}
    assert apply_features(engine,global_pose,compiled,.1,.3)['perturbation']=={'jitter':0.,'shimmer':100.,'seed':42}
    assert apply_features(engine,global_pose,compiled,.2,.3)['perturbation']==global_pose['perturbation']
    assert pose==original
    for key in ('jitter','shimmer'):
        with pytest.raises(ValueError):validate_global_controls(engine,{key:{'mode':'fixed','value':101}})
        with pytest.raises(ValueError):validate_global_controls(engine,{key:{'mode':'offset','value':float('nan')}})
    doc=make_document(engine,[pose],[],perturbation=pose['perturbation'],global_controls=controls,features=features)
    read=sequence_document(engine,doc)
    assert read['features']==features and read['global_controls']==controls and read['perturbation']['seed']==42

@pytest.mark.parametrize('parameter',['jitter','shimmer'])
def test_regional_source_changes_pcm_only_from_selected_boundary(engine,parameter):
    frames=[frame(engine)]
    baseline={'jitter':0.,'shimmer':0.,'seed':23}
    engine.configure_perturbation(baseline)
    dry=prepare_animation(engine,frames,pictures_enabled=False)['audio']
    features=[feature(parameter,80)]
    result=prepare_animation(engine,frames,features=features)
    again=prepare_animation(engine,frames,features=features,pictures_enabled=False)
    np.testing.assert_array_equal(result['audio'][:4800],dry[:4800])
    assert np.max(np.abs(result['audio'][4800:9600]-dry[4800:9600]))>1e-6
    np.testing.assert_array_equal(result['audio'],again['audio'])
    assert engine.perturbation==baseline
    states=[json.loads(zlib.decompress(p)) for p in result['pictures']]
    assert states[2]['perturbation'][parameter]==0
    assert states[3]['perturbation'][parameter]==80
    assert states[6]['perturbation'][parameter]==0

def test_cancel_restores_live_source(engine):
    baseline={'jitter':3.,'shimmer':4.,'seed':23};engine.configure_perturbation(baseline)
    cancel=threading.Event();block=engine.block_tube;count=0
    def interrupted(tube,g,n=960):
        nonlocal count
        result=block(tube,g,n);count+=n
        if count>=6000:cancel.set()
        return result
    engine.block_tube=interrupted
    with pytest.raises(ValueError,match='取消'):
        prepare_animation(engine,[frame(engine)],features=[feature('shimmer',80)],cancel=cancel)
    assert engine.perturbation==baseline
