from pathlib import Path
RESOURCES=Path(__file__).resolve().parents[2]/"resources/vocal_tract/native"
WEB=Path(__file__).resolve().parents[2]/"gui/resources/vocal_tract"
"""Native articulatory motion tests; only synthesize arrays, never play audio."""
import json
import zlib
import numpy as np
import pytest
from phonetic_toolbox.core.vocal_tract.engine import Engine, dp, ip, check
from phonetic_toolbox.services.vocal_tract.animation import prepare_animation
from phonetic_toolbox.core.vocal_tract.trajectory import sample_frame, validate_frames

@pytest.fixture(scope='module')
def engine():
    e=Engine(resource_dir=RESOURCES)
    yield e
    e.close()

def frames(engine):
    return [{'params':engine.presets[n],'lip_width':w,'f0':f0,'duration':.18}
            for n,w,f0 in [('a',.7,110),('i',1.3,170),('u',1.,130)]]

def test_lip_width_changes_native_lumen_and_transfer_without_changing_height(engine):
    a=engine.snapshot(engine.presets['a'],lip_width=.65)
    b=engine.snapshot(engine.presets['a'],lip_width=1.4)
    for name in ['upper_lip','lower_lip']:
        vertices=lambda s:np.array(next(m['vertices'] for m in s['meshes'] if m['name']==name)).reshape(-1,3)
        v,w=vertices(a),vertices(b)
        np.testing.assert_array_equal(v[:,:2],w[:,:2])
        assert np.ptp(w[:,2])>np.ptp(v[:,2])*1.3
    assert b['tube_areas'][-1]>a['tube_areas'][-1]
    assert np.max(np.abs(np.array(a['transfer_db'])-b['transfer_db']))>.2

def test_default_width_preserves_upstream_tube_geometry(engine):
    for name in engine.presets:
        p=engine.validated(engine.presets[name]);lengths=np.empty(50);areas=np.empty(50);arts=np.empty(50,dtype=np.int32);extras=[np.empty(1) for _ in range(3)]
        check(engine.analysis.vtlTractToTube(dp(p),dp(lengths),dp(areas),ip(arts),*[dp(a) for a in extras]))
        native=engine.prepare_tube(p)
        np.testing.assert_allclose(native['areas'],areas,atol=1e-8)
        np.testing.assert_allclose(native['lengths'],lengths,atol=1e-8)

@pytest.mark.parametrize('name',['a','o','u'])
def test_vowel_guard_limits_opening_instead_of_hiding_a_native_occlusion(engine,name):
    p=engine.preset(name);i=engine.names.index('VO');p[i]=1.5
    assert engine.prepare_tube(p)['areas'][16:].min()<.01
    safe,notice=engine.constrain_nasal_opening(p)
    assert notice and 0<safe[i]<p[i]
    np.testing.assert_array_equal(np.delete(safe,i),np.delete(p,i))
    assert engine.prepare_tube(safe)['areas'][16:].min()>=notice['minimum_oral_area']-1e-6

def test_trajectory_interpolates_all_controls_and_holds_final_pose(engine):
    sequence=frames(engine);first=sample_frame(sequence,0);middle=sample_frame(sequence,.09);end=sample_frame(sequence,.54)
    assert first['params']==sequence[0]['params']
    np.testing.assert_allclose(middle['params'],(np.array(sequence[0]['params'])+sequence[1]['params'])/2)
    assert middle['lip_width']==pytest.approx(1.) and middle['f0']==pytest.approx(140)
    for key in ['params','lip_width','f0']:assert end[key]==sequence[-1][key]
    for bad in [[],sequence*5,[{**sequence[0],'duration':float('nan')},sequence[1]],[{**f,'duration':3} for f in sequence*2]]:
        with pytest.raises(ValueError):validate_frames(engine,bad)

def test_audio_and_cached_view_use_same_trajectory(engine):
    sequence=frames(engine);result=prepare_animation(engine,sequence)
    assert len(result['audio'])==25920
    assert np.isfinite(result['audio']).all() and np.sqrt(np.mean(result['audio']**2))>.001
    assert result['times'][0]==0 and result['times'][-1]==pytest.approx(.54)
    for time,raw in zip(result['times'],result['pictures']):
        state=json.loads(zlib.decompress(raw));expected=sample_frame(sequence,time)
        np.testing.assert_allclose(state['params'],expected['params'])
        assert state['lip_width']==pytest.approx(expected['lip_width'])
        assert state['f0']==pytest.approx(expected['f0'])
        assert np.isfinite(state['tube_areas']).all()

def test_width_changes_synthesized_audio(engine):
    signals=[]
    for width in [.65,1.4]:
        sequence=[{'params':engine.presets['a'],'lip_width':width,'f0':125,'duration':.2}]*2
        signals.append(prepare_animation(engine,sequence,pictures_enabled=False)['audio'])
    assert np.sqrt(np.mean((signals[0]-signals[1])**2))>.001
