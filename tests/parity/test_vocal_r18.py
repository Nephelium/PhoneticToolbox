"""R18 native shape, portable documents and pose-free aspiration intervals."""
from pathlib import Path
import copy
import json
import hashlib
import numpy as np
import pytest
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.shape import validate_tongue_shape
from phonetic_core.vocal_tract.trajectory import validate_frames,sample_frame
from phonetic_core.vocal_tract.animation import prepare_animation
from phonetic_core.vocal_tract.document import make_document,sequence_document
from ptb_desktop.vocal_tract.profile import ProfileStore,DEFAULT_FAVORITES

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def engine():
    e=Engine(resource_dir=ROOT/'resources/vocal_tract/native')
    yield e
    e.close()

def frame(e,name='a',**kw):
    return dict(params=e.presets[name],duration=.1,f0=150,lip_width=1,**kw)

def test_native_shape_changes_mesh_and_acoustics_without_state_leak(engine):
    p=engine.presets['a'];before=engine.snapshot(p)
    altered=engine.snapshot(p,tongue_shape=[.1,.35,0,0])
    assert altered['meshes'][6]['vertices']!=before['meshes'][6]['vertices']
    assert altered['tube_areas']!=before['tube_areas']
    assert altered['transfer_db']!=before['transfer_db']
    assert altered['contours']['lower_cover']==before['contours']['lower_cover']
    assert 0<altered['tongue_surface_rib']<altered['tongue_blade_rib']
    tube=engine.prepare_tube(p,tongue_shape=[.1,.35,0,0])
    assert np.allclose(tube['areas'],altered['tube_areas'],atol=5e-7,rtol=0)
    assert engine.snapshot(p)['meshes']==before['meshes']
    assert np.allclose(engine.prepare_tube(p)['areas'],before['tube_areas'],atol=5e-7,rtol=0)

@pytest.mark.parametrize('bad',[[0]*3,[0,0,0,float('nan')],[0,0,0,.41],[0,-.81,0,0],True])
def test_shape_bounds(bad):
    with pytest.raises(ValueError):validate_tongue_shape(bad)

def test_aspiration_preserves_continuous_motion(engine):
    a=frame(engine,tongue_shape=[0]*4);b=frame(engine,'i',tongue_shape=[.2,.4,-.1,-.2])
    sequence=validate_frames(engine,[a,{'aspiration':True,'duration':.01},b]);original=copy.deepcopy(sequence)
    for time in [.05,.099999,.1,.105,.109999,.11,.15]:
        p=sample_frame(sequence,time)
        t=np.clip(time/.11,0,1);u=t*t*(3-2*t)
        assert np.allclose(p['params'],(1-u)*np.array(a['params'])+u*np.array(b['params']))
        assert np.allclose(p['tongue_shape'],u*np.array(b['tongue_shape']))
        assert bool(p.get('aspiration'))==(.1<=time<.11)
        if p.get('aspiration'):assert p['source']['mode']=='voiceless' and p['source']['vibration']==0 and not p.get('silent')
    assert sequence==original

def test_aspiration_leading_trailing_and_multiple(engine):
    a=frame(engine);asp={'aspiration':True,'duration':.02}
    seq=validate_frames(engine,[asp,asp,a,asp])
    for t in [0,.03,.14,.159]:assert sample_frame(seq,t)['params']==a['params']
    with pytest.raises(ValueError,match='构形帧'):validate_frames(engine,[asp])
    for bad in [{**asp,'silent':True},{**asp,'params':a['params']},{**asp,'duration':.009}]:
        with pytest.raises(ValueError):validate_frames(engine,[a,bad])

def test_native_10ms_aspiration_is_bounded_at_exact_samples(engine):
    sequence=[frame(engine),{'aspiration':True,'duration':.01},frame(engine,'i')]
    calls=[];original=engine.block_tube
    def record(tube,g,n=960):
        calls.append((n,float(g[engine.gnames.index('rel_amp')]),float(g[1]),tube['areas'].copy()))
        return original(tube,g,n)
    engine.block_tube=record
    try:result=prepare_animation(engine,sequence,pictures_enabled=False)
    finally:engine.block_tube=original
    position=0;unvoiced=[]
    for n,vibration,pressure,areas in calls:
        if n and vibration==0:unvoiced.append((position,position+n));assert pressure>0
        position+=n
    zero_updates=[areas for n,vibration,pressure,areas in calls if n==0]
    assert len(zero_updates)==3
    for time,areas in zip([.1,.11],zero_updates[1:]):
        pose=sample_frame(validate_frames(engine,sequence),time)
        expected=engine.prepare_tube(pose['params'],pose['lip_width'],pose.get('larynx_height',0),pose.get('tongue_shape'))['areas']
        assert np.array_equal(areas,expected)
    assert unvoiced==[(4800,5280)]
    assert position==10080==len(result['audio']) and np.isfinite(result['audio']).all()
    assert np.std(result['audio'][4800:5280])>0

def test_file_v5_and_legacy_loading(engine):
    frames=[frame(engine,tongue_shape=[.1,.2,-.1,-.2]),{'aspiration':True,'duration':.01,'name':'ʰ（短）'},frame(engine,'i')]
    doc=make_document(engine,frames,[])
    assert doc['version']==8 and doc['geometry_method']=='m10/r21'
    assert sequence_document(engine,doc)==doc
    assert doc['frames'][1]['name']=='ʰ（短）'
    with pytest.raises(ValueError,match='名称'):validate_frames(engine,[frame(engine),{'aspiration':True,'duration':.01,'name':'bad\nname'}])
    for version,model,method in [(1,'VTL-2.4-JD2',None),(2,'VTL-2.4-JD2-M10-3',None),(3,'VTL-2.4-JD2-M10-R16','m10/r16'),(4,'VTL-2.4-JD2-M10-R17','m10/r17')]:
        old={**doc,'version':version,'model':model,'geometry_method':method,'frames':[frame(engine)]}
        read=sequence_document(engine,old);assert 'R21' in read['geometry_notice']
        assert read['frames'][0]['params']==old['frames'][0]['params']
        with pytest.raises(ValueError,match='版本 5'):sequence_document(engine,{**old,'frames':frames})

def test_favorites_and_local_persistence(engine,tmp_path):
    profile=ProfileStore(tmp_path);assert profile.load_favorites()==DEFAULT_FAVORITES
    custom={**frame(engine,tongue_shape=[.1,.2,0,.1]),'name':'t͡s','id':'qa-custom'}
    profile.save_presets(engine,[custom]);profile.save_favorites(engine,['s','qa-custom'])
    fresh=ProfileStore(tmp_path);assert fresh.load_favorites()==['s','qa-custom']
    assert fresh.load_presets(engine)[0]['tongue_shape']==custom['tongue_shape']
    fresh.save_favorites(engine,[]);assert fresh.load_favorites()==[]
    with pytest.raises(ValueError):fresh.save_favorites(engine,['missing'])
    seq=[frame(engine),{'aspiration':True,'duration':.01}]
    fresh.save_frames(validate_frames(engine,seq));assert fresh.load_frames(engine)[1]==seq[1]
