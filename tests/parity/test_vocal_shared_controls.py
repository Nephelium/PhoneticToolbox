"""Shared authoring, model isolation and the reported boundary-preview regression."""
from pathlib import Path
import copy
import json
import os
import zlib
import numpy as np
import pytest
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.animation import prepare_animation
from phonetic_core.vocal_tract.document import make_document, sequence_document
from phonetic_core.vocal_tract.trajectory import sample_frame, validate_frames
from phonetic_core.vocal_tract.shape import TONGUE_SHAPE_LIMITS, validate_tongue_shape
from phonetic_core.vocal_tract.source import validate_source
from phonetic_core.vocal_tract.sequence_controls import ipa_sequence, validate_global_controls, apply_global_controls
from phonetic_core.vocal_tract.session import Runtime
from ptb_worker.vocal_web import MemoryProfile, BrowserAudio
from ptb_worker.browser_vocal import BrowserVocalRuntime
from ptb_desktop.vocal_tract.profile import ProfileStore

ROOT=Path(__file__).resolve().parents[2]
RESOURCES=Path(os.environ.get('PTB_M10_TEST_RESOURCES',ROOT/'resources/vocal_tract/native'))

@pytest.fixture
def engine():
    e=Engine(resource_dir=RESOURCES)
    yield e
    e.close()


def test_sixty_second_sequence_budget_and_complete_native_buffers(engine):
    frames=[{**pose(engine),'duration':3.,'f0':engine.default_f0} for _ in range(20)]
    result=prepare_animation(engine,frames)
    assert result['duration']==60. and len(result['audio'])==60*engine.sr
    assert len(result['pictures'])==1801 and result['times'][-1]==60.
    assert np.isfinite(result['audition_audio']).all() and np.max(np.abs(result['audio']))>0
    with pytest.raises(ValueError,match='60 秒'):
        validate_frames(engine,frames+[pose(engine)])
    # A longer authored document remains storable for splitting into selections.
    assert len(validate_frames(engine,frames+[pose(engine)],for_storage=True))==21

def pose(e,name='a',**kw):
    return dict(params=e.presets[name],name=name,f0=150.,lip_width=1.,duration=.1,**kw)

def point(state,key):
    r=state[key];lo=int(r);t=r-lo;p=np.asarray(state['contours']['tongue'])
    return p[lo]*(1-t)+p[lo+1]*t

def test_every_boundary_interpolates_without_a_single_ulp_overflow(engine):
    lo=[a for a,b in TONGUE_SHAPE_LIMITS];hi=[b for a,b in TONGUE_SHAPE_LIMITS]
    for start,end in [(lo,lo),(hi,hi),(lo,hi),(hi,lo)]:
        a=pose(engine,tongue_shape=start,larynx_height=-1.,source={'pressure_pa':1600.})
        b=pose(engine,tongue_shape=end,larynx_height=1.,source={'pressure_pa':1600.})
        a['lip_width']=b['lip_width']=1.6
        for t in np.linspace(0,.1,1001):
            p=sample_frame([a,b],float(t));validate_tongue_shape(p['tongue_shape']);validate_source(p['source'])
            assert .55<=p['lip_width']<=1.6 and -1<=p['larynx_height']<=1
    # Exact user path: belly reaches 0.4, followed by short preview (two equal poses).
    r=Runtime(engine,MemoryProfile(),BrowserAudio(engine))
    r.invoke('pose',dict(params=engine.presets['a'],tongue_shape=[0,0,0,.4],revision=1))
    r.invoke('preview',{'duration':.5})
    raw=r.live.buffer
    assert len(raw)==24000 and np.isfinite(raw).all() and np.max(np.abs(raw))>0

def test_ipa_longest_saved_symbol_unicode_and_atomic_missing(engine):
    catalog=[pose(engine),pose(engine,'i'),pose(engine,name='u'),{**pose(engine),'name':'t͡s'},
             {**pose(engine),'name':'ã'}]
    source=copy.deepcopy(catalog)
    value=ipa_sequence(engine,'[at͡si a\u0303]',catalog,.25)
    assert value['symbols']==['a','t͡s','i','ã'] and not value['missing']
    assert all(p['duration']==.25 for p in value['frames'])
    value['frames'][0]['params'][0]+=.01
    assert catalog==source
    for text,missing in [('atʰi','tʰ'),('aːi','aː'),('t͡ʃi','t͡ʃ'),('ˈai','ˈ')]:
        value=ipa_sequence(engine,text,catalog)
        assert not value['frames'] and missing in value['missing']
    catalog.append({**pose(engine),'name':'a','f0':211.})
    assert ipa_sequence(engine,'a',catalog)['frames'][0]['f0']==211

def test_global_fixed_relative_clamp_and_original_is_unchanged(engine):
    frame=pose(engine);before=copy.deepcopy(frame)
    controls=validate_global_controls(engine,{'VO':{'mode':'fixed','value':0},'TCY':{'mode':'offset','value':float(engine.maximum[9]-engine.minimum[9])},'larynx_height':{'mode':'fixed','value':.7},'lip_width':{'mode':'offset','value':.9}})
    actual=apply_global_controls(engine,frame,controls)
    assert actual['params'][7]==0 and actual['params'][9]==engine.maximum[9]
    assert actual['lip_width']==1.6 and actual['larynx_height']==.7
    assert set(actual['global_limited'])=={'lip_width','TCY'} and frame==before
    for bad in [{'X':{'mode':'fixed','value':0}},{'VO':{'mode':'offset','value':True}},{'VO':{'mode':'fixed','value':-1}},{'VO':{'mode':'offset','value':float('nan')}}]:
        with pytest.raises(ValueError):validate_global_controls(engine,bad)


def test_ipa_durations_follow_manner_length_and_explicit_override(engine):
    expected={'a':.2,'ã':.2,'aː':.4,'iˑ':.3,'p':.08,'b':.08,'tʰ':.12,
              't͡s':.14,'d͜ʒ':.14,'s':.18,'z':.16,'n':.12,'l':.10,'ɾ':.06}
    catalog=[{**pose(engine),'name':symbol} for symbol in expected]
    result=ipa_sequence(engine,' '.join(expected),catalog)
    assert [f['duration'] for f in result['frames']]==list(expected.values())
    assert all(f['duration']==.27 for f in ipa_sequence(engine,' '.join(expected),catalog,.27)['frames'])
    assert all(f['duration']==.1 for f in catalog)
    for bad in [True,0,.049,3.1,float('nan')]:
        with pytest.raises(ValueError):ipa_sequence(engine,'a',catalog,bad)

def test_global_picture_audio_and_cache_use_same_pose(engine):
    r=Runtime(engine,MemoryProfile(),BrowserAudio(engine));frames=[pose(engine),pose(engine,'i')];before=copy.deepcopy(frames)
    controls={'VO':{'mode':'fixed','value':0},'larynx_height':{'mode':'fixed','value':.3}}
    a=r.invoke('animation/prepare',dict(frames=frames,global_controls=controls))
    assert frames==before and not a['cached']
    assert r.invoke('animation/prepare',dict(frames=frames,global_controls=controls))['cached']
    for index in [0,a['count']-1]:
        state=r.invoke('animation/picture',{'id':a['id'],'index':index})['state']
        assert state['params'][7]==0 and state['larynx_height']==.3
        assert np.allclose(engine.prepare_tube(state['params'],state['lip_width'],.3,state['tongue_shape'])['areas'],state['tube_areas'],atol=5e-7,rtol=0)
    assert not r.invoke('animation/prepare',dict(frames=frames,global_controls={}))['cached']

def test_models_profiles_checkpoints_and_portable_files(engine,tmp_path):
    for profile in [MemoryProfile(),ProfileStore(tmp_path)]:
        r=Runtime(engine,profile,BrowserAudio(engine))
        item={**pose(engine),'name':'t͡s','id':'custom'}
        r.invoke('presets/save',{'presets':[item]});r.invoke('keyframes',{'frames':[pose(engine)]})
        r.invoke('speaker/select',{'speaker':'W02'})
        assert engine.speaker=='W02' and engine.default_f0==200
        assert r.invoke('presets/load')==[] and r.invoke('keyframes/load')['frames']==[]
        controls={'larynx_height':{'mode':'fixed','value':.5}}
        doc=make_document(engine,[pose(engine)],[],global_controls=controls)
        assert doc['version']==8 and doc['speaker']=='W02' and sequence_document(engine,doc)==doc
        r.invoke('keyframes',{'frames':doc['frames'],'global_controls':controls})
        r.invoke('speaker/select',{'speaker':'JD2'})
        assert r.invoke('presets/load')[0]['id']=='custom'
        assert len(r.invoke('keyframes/load')['frames'])==1
        with pytest.raises(ValueError,match='另一声道模型'):sequence_document(engine,doc)
        old={'format':doc['format'],'version':5,'model':'VTL-2.4-JD2-M10-R18','geometry_method':'m10/r18','parameters':engine.names,'frames':[pose(engine)],'pitch_curve':[]}
        read=sequence_document(engine,old);assert read['frames'][0]['params']==old['frames'][0]['params']
        assert 'R21' in read['geometry_notice']
        r.invoke('speaker/select',{'speaker':'W02'})
        assert r.invoke('keyframes/load')['global_controls']==controls
        r.invoke('speaker/select',{'speaker':'JD2'})

def test_native_surface_moves_down_and_remains_finite(engine):
    assert engine.geometry_method=='m10/r21' and engine.metadata()['source_perturbation']
    for name in ['a','i','u']:
        base=engine.snapshot(engine.presets[name]);down=engine.snapshot(engine.presets[name],tongue_shape=[0,-.8,0,0])
        assert point(down,'tongue_surface_rib')[1]<point(base,'tongue_surface_rib')[1]-.65
        assert down['tube_areas']!=base['tube_areas'] and np.isfinite(down['tube_areas']).all()
        p=np.asarray(engine.presets[name]).copy();p[13]=base['limited'][13]-.7
        blade=engine.snapshot(p)
        assert point(blade,'tongue_blade_rib')[1]<point(base,'tongue_blade_rib')[1]-.2


def test_closed_tract_prevoicing_survives_before_oral_release(engine):
    # A held closed pose followed by release. Closure is not explicit silence.
    for symbol in ['b','d']:
        p=list(engine.presets['a' if symbol=='b' else 't']);p[7]=0
        if symbol=='b':p[5]=-1
        assert min(engine.prepare_tube(p)['areas'])<=.00011
        held=dict(params=p,duration=.2,f0=150,source={'mode':'voiced'})
        release={**held,'duration':.05}
        vowel={**pose(engine),'duration':.2}
        y=prepare_animation(engine,[held,release,vowel],pictures_enabled=False)['audio']
        lead=y[960:8640];tail=y[-4800:-960]
        assert np.sqrt(np.mean(lead**2))>.001
        assert np.corrcoef(lead[:-320],lead[320:])[0,1]>.9
        assert np.sqrt(np.mean(lead**2))<np.sqrt(np.mean(tail**2))*.3
        quiet=prepare_animation(engine,[{**held,'silent':True},release,vowel],pictures_enabled=False)['audio']
        assert not np.any(quiet[:9600])


def test_f0_compensation_preserves_pressure_and_perturbation(engine):
    r=Runtime(engine,MemoryProfile(),BrowserAudio(engine))
    for speaker,reference in [('JD2',120),('W02',200)]:
        r.invoke('speaker/select',{'speaker':speaker})
        assert engine.default_f0==reference
        engine.configure_perturbation({'jitter':0,'shimmer':0,'seed':1})
        for vowel in ['a','i','u']:
            tube=engine.prepare_tube(engine.presets[vowel]);levels=[]
            for f0 in [60,80,120,200,300,350]:
                g=engine.glottis(f0,source={'mode':'voiced'});gain=engine.pitch_level_gain(tube,f0)
                engine.reset_tube(tube,g);raw=engine.block_tube(tube,g,24000)[9600:]
                levels.append(float(np.sqrt(np.mean((raw*gain)**2))))
                assert gain==engine.pitch_level_gain(tube,f0,{'pressure_pa':400})
                assert engine.pitch_level_gain(tube,f0,{'mode':'voiceless'})==1
            assert 20*np.log10(max(levels)/min(levels))<1.0
        # Reference calculation must not reset synthesis or shimmer state.
        p=engine.presets['a'];tube=engine.prepare_tube(p);g=engine.glottis(150)
        engine.configure_perturbation({'jitter':12,'shimmer':35,'seed':7})
        engine.reset_tube(tube,g);expected=np.concatenate([engine.block_tube(tube,g,4800),engine.block_tube(tube,g,4800)])
        engine.reset_tube(tube,g);first=engine.block_tube(tube,g,4800)
        engine.pitch_level_cache.clear();engine.pitch_level_gain(tube,211)
        second=engine.block_tube(tube,g,4800)
        np.testing.assert_allclose(np.concatenate([first,second]),expected,atol=1e-12,rtol=0)


def test_portable_preset_pack_roundtrip_and_model_boundaries(engine):
    from phonetic_core.vocal_tract.preset_pack import make_pack,validate_pack
    catalog=[{**pose(engine),'name':'/ã/','source':{'mode':'voiced'},'id':'private-id'},
             {**pose(engine,'i'),'name':'t͡s','tongue_shape':[.1,-.5,0,.2]}]
    original=copy.deepcopy(catalog)
    pack=make_pack(engine,{'title':'测试音标','presets':catalog})
    assert validate_pack(engine,json.loads(json.dumps(pack)))==pack
    assert [p['name'] for p in pack['presets']]==['ã','t͡s']
    assert all('id' not in p for p in pack['presets']) and catalog==original
    assert ipa_sequence(engine,'ãt͡s',pack['presets'])['symbols']==['ã','t͡s']
    for change in [{'speaker':'W02'},{'speaker_sha256':'invalid'},{'parameters':engine.names[::-1]},
                   {'geometry_method':'future'},{'version':99},{'presets':[]},
                   {'presets':[pack['presets'][0],pack['presets'][0]]}]:
        with pytest.raises(ValueError):validate_pack(engine,{**pack,**change})
