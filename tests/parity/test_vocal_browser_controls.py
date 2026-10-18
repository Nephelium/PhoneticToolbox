"""Independent timeline expectations and portable authoring compatibility."""
import copy
from pathlib import Path
import numpy as np
import pytest
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.trajectory import validate_frames,validate_pressure_curve,sample_trajectory
from phonetic_core.vocal_tract.document import make_document,sequence_document
from phonetic_core.vocal_tract.animation import prepare_animation
from phonetic_core.vocal_tract.perturbation import validate_perturbation
from ptb_worker.vocal_web import MemoryProfile
from ptb_desktop.vocal_tract.profile import ProfileStore

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def engine():
    e=Engine(resource_dir=ROOT/'resources/vocal_tract/native')
    yield e
    e.close()


def frames(e):return validate_frames(e,[dict(params=e.presets['a'],duration=.2,lip_width=1.,f0=150)])


def test_pressure_is_linear_pa_and_independent_of_f0_and_input(engine):
    seq=frames(engine);before=copy.deepcopy(seq);pressure=[[0,200],[1,1400]]
    p=sample_trajectory(seq,.05,[[0,100],[1,300]],pressure)
    assert p['f0']==150 and p['source']['pressure_pa']==500
    assert p['source']['audition_gain_db']==0 and p['params']==seq[0]['params']
    assert seq==before and pressure==[[0,200],[1,1400]]
    assert sample_trajectory(seq,.05)==sample_trajectory(seq,.05,[],[])


def test_aspiration_accepts_pressure_silence_is_not_unmuted(engine):
    seq=frames(engine);seq=[{**seq[0],'duration':.1},{'aspiration':True,'duration':.05},{**seq[0],'duration':.1,'silent':True}]
    p=sample_trajectory(seq,.125,[[0,100],[1,300]],[[0,300],[1,1300]])
    assert p['aspiration'] and p['source']['vibration']==0 and p['source']['pressure_pa']==800
    quiet=sample_trajectory(seq,.2,[[0,100],[1,300]],[[0,300],[1,1300]])
    assert quiet['silent'] and 'source' not in quiet
    result=prepare_animation(engine,seq,pictures_enabled=False,pressure_curve=[[0,300],[1,1300]])
    assert np.isfinite(result['audio']).all() and not np.any(result['audio'][7200:])


@pytest.mark.parametrize('curve',[None,[[0,1]],[[.1,0],[1,100]],[[0,0],[1,1601]],[[0,-1],[1,10]],[[0,0],[0,10],[1,100]],[[0,0],[1,float('nan')]]])
def test_invalid_pressure_rejected(curve):
    with pytest.raises((ValueError,TypeError)):validate_pressure_curve(curve)


def test_v5_unchanged_v6_explicit_and_both_profiles_roundtrip(engine,tmp_path):
    seq=frames(engine);curve=[[0,200],[1,1400]];p={'jitter':12.,'shimmer':25.,'seed':17}
    # Explicit historical documents keep their original schema and geometry.
    current=make_document(engine,seq,[])
    assert current['version']==8 and sequence_document(engine,current)==current
    old={k:v for k,v in current.items() if k not in ('speaker','global_controls','pressure_curve','perturbation')}
    old.update(version=5,model='VTL-2.4-JD2-M10-R18',geometry_method='m10/r18')
    assert old['version']==5 and 'pressure_curve' not in old and 'perturbation' not in old
    assert sequence_document(engine,old)['frames']==seq
    for pressure,perturb in [(curve,None),([],p),(curve,p)]:
        doc=make_document(engine,seq,[],pressure,perturb)
        assert doc['version']==8 and sequence_document(engine,doc)==doc
        assert doc['pressure_curve']==pressure and doc['perturbation']==perturb
        legacy={k:v for k,v in doc.items() if k not in ('speaker','global_controls')}
        legacy.update(version=6,model='VTL-2.4-JD2-M10-R18',geometry_method='m10/r18')
        read=sequence_document(engine,legacy)
        assert read['frames']==seq and read['pressure_curve']==pressure and read['perturbation']==perturb
        if perturb is not None:
            with pytest.raises(ValueError,match='抖动方法'):sequence_document(engine,{**doc,'perturbation_method':'unknown'})
        with pytest.raises(ValueError,match='版本 6'):sequence_document(engine,{**doc,'version':5})
    for profile in [MemoryProfile(),ProfileStore(tmp_path)]:
        profile.save_frames(seq,[],curve,p)
        restored=profile.load_sequence(engine)
        assert restored['pressure_curve']==curve and restored['perturbation']==p
        profile.save_frames(seq,[])
        assert profile.load_sequence(engine)=={'frames':seq,'pitch_curve':[]}


@pytest.mark.parametrize('value',[{'jitter':-1},{'shimmer':101},{'jitter':True},{'seed':1.2},{'seed':-1},{'seed':True},{'seed':4294967296},{'unknown':3}])
def test_invalid_perturbation_rejected(value):
    with pytest.raises(ValueError):validate_perturbation(value)


def test_custom_source_requires_a_capable_engine(engine):
    assert validate_perturbation() is None
    assert validate_perturbation({})==dict(jitter=10.,shimmer=0.,seed=1)
    if not engine.metadata()['source_perturbation']:
        with pytest.raises(ValueError,match='不支持自定义'):engine.configure_perturbation({})


def test_source_update_does_not_recompute_geometry_and_keeps_the_live_tube(engine,monkeypatch):
    from phonetic_core.vocal_tract.session import Runtime
    from ptb_worker.vocal_web import BrowserAudio
    audio=BrowserAudio(engine);r=Runtime(engine,MemoryProfile(),audio)
    r.invoke('pose',dict(params=engine.presets['a'],revision=1))
    before=audio.params.copy();audio.tube=engine.prepare_tube(audio.params);tube=audio.tube
    def forbidden(*args,**kwargs):raise AssertionError('Source update must not compute geometry')
    monkeypatch.setattr(engine,'snapshot',forbidden)
    assert r.invoke('source/update',dict(revision=2,f0=211,source={'pressure_pa':900}))['f0']==211
    assert audio.f0==211 and audio.source['pressure_pa']==900 and audio.tube is tube
    np.testing.assert_array_equal(audio.params,before)
    with pytest.raises(ValueError,match='过期'):r.invoke('source/update',dict(revision=2,f0=200))
    with pytest.raises(ValueError):r.invoke('source/update',dict(revision=3,f0=500))
    assert r.revision==2 and audio.f0==211


def test_stationary_frame_reuses_geometry_while_retaining_every_display_time(engine,monkeypatch):
    counts={'tube':0,'snapshot':0};tube=engine.prepare_tube;snapshot=engine.snapshot
    def counted_tube(*a,**kw):counts['tube']+=1;return tube(*a,**kw)
    def counted_snapshot(*a,**kw):counts['snapshot']+=1;return snapshot(*a,**kw)
    monkeypatch.setattr(engine,'prepare_tube',counted_tube);monkeypatch.setattr(engine,'snapshot',counted_snapshot)
    a=prepare_animation(engine,frames(engine),pressure_curve=[[0,500],[1,1000]])
    assert counts==dict(tube=1,snapshot=1)
    assert len(a['pictures'])==7 and a['times']==[i/30 for i in range(7)]
    assert len(a['audio'])==9600 and np.isfinite(a['audio']).all()
