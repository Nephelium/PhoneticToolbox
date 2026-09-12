"""M10 numeric/behavior regression; native VTL required, no soundcard access."""
from pathlib import Path
import threading
import numpy as np
import pytest
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.trajectory import validate_frames,sample_frame
from phonetic_core.vocal_tract.animation import prepare_animation

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def engine():
    e=Engine(resource_dir=ROOT/'resources/vocal_tract/native')
    yield e
    e.close()

def frame(e,preset='a',**kw):
    return dict(params=e.presets[preset],f0=125,lip_width=1,duration=.2,**kw)

def test_vtl24_consonant_areas_from_old_native_capture(engine):
    engine.set_manual_root(False)
    engine.geom.p3_lateral_fit(0)
    try:
        for name,area in [('lateral',.2),('fricative',.15),('closure',.0001)]:
            state=engine.snapshot(engine.preset('tt-alveolar-'+name+'(a)'))
            assert state['oral_min_area']==pytest.approx(area,abs=1e-10)
            assert state['nasal']['port_area']==0
    finally:engine.geom.p3_lateral_fit(1)

def test_lateral_has_real_central_contact_and_two_side_passages(engine):
    state=engine.snapshot(engine.presets['l'],section=109)
    up=np.array([0 if v is None else v for v in state['upper']])
    lo=np.array([0 if v is None else v for v in state['lower']])
    gap=up-lo
    assert gap[48]<.001
    assert gap[:47].sum()>.1 and gap[49:].sum()>.1
    assert state['oral_min_area']==pytest.approx(.2,abs=1e-9)
    assert engine.snapshot(engine.presets['s'])['oral_min_area']==pytest.approx(.09,abs=1e-9)

def test_full_closure_is_near_silent_without_output_gating(engine):
    engine.set_manual_root(False)
    def rms(symbol):
        tube=engine.prepare_tube(engine.presets[symbol]);g=engine.glottis(source={'mode':'voiceless'})
        engine.reset_tube(tube,g)
        # Use the captured one-second condition, after the startup ring-down.
        audio=np.concatenate([engine.block_tube(tube,g) for _ in range(50)])
        assert np.isfinite(audio).all()
        return np.std(audio[-20000:])
    assert rms('t')<1e-8
    assert rms('s')>1e-4

def test_nasal_opening_keeps_path_even_with_oral_closure(engine):
    state=engine.snapshot(engine.presets['n'])
    assert state['nasal']['port_area']==pytest.approx(.5)

def test_manual_root_changes_real_geometry_and_auto_remains_default(engine):
    p=np.array(engine.presets['a']);i=engine.names.index('TRX');p[i]-=.5
    engine.set_manual_root(False);a=engine.snapshot(p)
    engine.set_manual_root(True);b=engine.snapshot(p)
    assert np.max(np.abs(np.array(a['tube_areas'])-b['tube_areas']))>.05
    assert a['limited'][i]!=pytest.approx(b['limited'][i])
    engine.set_manual_root(False);c=engine.snapshot(p)
    np.testing.assert_array_equal(a['tube_areas'],c['tube_areas'])

def test_unpaged_frames_names_and_legacy(engine):
    frames=validate_frames(engine,[frame(engine,name='/n/ 舌尖',manual_root=True)]*45,for_storage=True)
    assert len(frames)==45 and frames[-1]['name']=='/n/ 舌尖'
    legacy=validate_frames(engine,[frame(engine)],for_storage=True)
    assert legacy[0]['manual_root'] is False and legacy[0]['name']==''
    with pytest.raises(ValueError):validate_frames(engine,[frame(engine,name='x\ny')],for_storage=True)
    with pytest.raises(ValueError):validate_frames(engine,[frame(engine)]*70)

def test_animation_cancelled_before_rendering(engine):
    cancel=threading.Event();cancel.set()
    with pytest.raises(ValueError,match='取消'):prepare_animation(engine,[frame(engine),frame(engine,'i')],cancel=cancel)

def test_names_are_not_acoustic_parameters(engine):
    a=[frame(engine,name='/n/'),frame(engine,'i',name='/a/')]
    b=[frame(engine,name='changed'),frame(engine,'i',name='changed')]
    assert sample_frame(a,.1)==sample_frame(b,.1)
