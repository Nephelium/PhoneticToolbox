"""M10-R11 native geometry, persistence and synchronized synthesis contracts."""
import json
import zlib
from pathlib import Path
import numpy as np
import pytest
from phonetic_core.vocal_tract.document import make_document,sequence_document
from phonetic_core.vocal_tract.trajectory import validate_frames,sample_frame
from phonetic_core.vocal_tract.animation import prepare_animation
from ptb_desktop.vocal_tract.runtime import Runtime

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime(tmp_path):
    runtime=Runtime(ROOT/'resources/vocal_tract/native',tmp_path/'profile',playback_allowed=False)
    yield runtime
    runtime.close()


def test_tongue_tip_can_cross_incisors_and_lips_without_rising(runtime):
    e=runtime.engine;p=e.presets['a'].copy();x=e.names.index('TTX');y=e.names.index('TTY')
    previous=-100
    for target in np.linspace(4.5,7.2,28):
        p[x]=target;s=e.snapshot(p)
        assert s['limited'][x]==pytest.approx(target,abs=1e-8)
        assert s['limited'][y]==pytest.approx(p[y],abs=1e-8)
        tip=max(q[0] for q in s['contours']['tongue']);assert tip>previous;previous=tip
    assert tip>max(q[0] for q in s['contours']['upper_lip'])+.2
    assert max(s['tube_areas'])<30 and np.isfinite(s['transfer_db']).all()


def test_blade_is_not_limited_by_old_construction_cones(runtime):
    e=runtime.engine;p=e.presets['a'].copy();p[12]=3.9;p[13]=-1.05
    s=e.snapshot(p)
    assert s['limited'][12:14]==pytest.approx([3.9,-1.05],abs=1e-8)
    assert all(np.isfinite(m['vertices']).all() for m in s['meshes'])


def test_independent_larynx_changes_geometry_and_acoustics_not_hyoid(runtime):
    e=runtime.engine;p=e.presets['a'];base=e.snapshot(p);length=sum(base['tube_lengths'])
    for shift in [-1,-.5,.5,1]:
        s=e.snapshot(p,larynx_height=shift);tube=e.prepare_tube(p,larynx_height=shift)
        assert s['params']==base['params']
        assert s['contours']['lower_cover'][4]==base['contours']['lower_cover'][4]
        assert s['contours']['lower_cover'][0][1]-base['contours']['lower_cover'][0][1]==pytest.approx(shift,abs=1e-5)
        assert (sum(s['tube_lengths'])-length)*shift<0
        np.testing.assert_allclose(tube['areas'],s['tube_areas'],atol=5.1e-7,rtol=0)
        assert not np.allclose(s['transfer_db'],base['transfer_db'])
        for name in ['upper_cover','lower_cover']:
            mesh=next(m for m in s['meshes'] if m['name']==name)
            xyz=np.array(mesh['vertices']).reshape(mesh['ribs'],mesh['points'],3)
            np.testing.assert_allclose(xyz[:,mesh['points']//2,:2],s['contours'][name],atol=1e-5,rtol=0)
            np.testing.assert_allclose(xyz[:,:,:2],xyz[:,::-1,:2],atol=1e-5,rtol=0)
    np.testing.assert_array_equal(e.snapshot(p)['tube_areas'],base['tube_areas'])


@pytest.mark.parametrize('bad',[True,float('nan'),float('inf'),-1.01,1.01])
def test_larynx_range_is_checked(runtime,bad):
    with pytest.raises(ValueError):runtime.engine.snapshot(runtime.engine.presets['a'],larynx_height=bad)
    with pytest.raises(ValueError):validate_frames(runtime.engine,[{'params':runtime.engine.presets['a'],'larynx_height':bad}],for_storage=True)


def test_fully_closed_cross_section_is_not_reopened_by_area_correction(runtime):
    e=runtime.engine;checked=0
    # Actual full-width lip closure. The former posterior fixtures depended on
    # tissue interpenetration and must not be retained as valid contact cases.
    for name,opening,expected in [('a',-2,125),('a',-1,125),('i',-2,122),('u',-1,124),('t',-1,124)]:
        p=e.presets[name].copy();p[5]=opening;s=e.snapshot(p)
        expected=round(expected*(e.section_count-1)/128)
        assert s['airway_sections'][expected]['area']==0
        lengths=np.cumsum(s['tube_lengths'])
        for i,sec in enumerate(s['airway_sections']):
            gaps=[a-b for a,b in zip(sec['upper'],sec['lower']) if a is not None and b is not None]
            if i>16 and not any(g>1e-6 for g in gaps):
                assert sec['area']==0
                assert s['raw_areas'][i]==0
                station=min(len(lengths)-1,int(np.searchsorted(lengths,s['centerline'][i][2])))
                # The acoustic solver retains its finite numerical closure floor.
                assert s['tube_areas'][station]<=.00011
                checked+=1
    assert checked>=5


def test_midline_contact_retains_real_lateral_air_passage(runtime):
    e=runtime.engine;s=e.snapshot(e.presets['l'])
    station=round(109*(e.section_count-1)/128)
    sec=s['airway_sections'][station]
    assert sec['upper'][48] is None or sec['upper'][48]<=sec['lower'][48]
    assert sec['area']>.01 and s['raw_areas'][station]>.01


def test_uvula_stops_at_tongue_without_moving_the_tongue(runtime):
    e=runtime.engine;p=e.presets['a'].copy();p[7]=1.5;s=e.snapshot(p)
    assert .6<s['uvula_contact_lift']<.7
    assert s['params']==list(p)
    tongue=np.asarray(s['contours']['tongue'][:34])
    uvula=next(m for m in s['meshes'] if m['name']=='uvula')
    front=np.asarray(uvula['vertices']).reshape(uvula['ribs'],uvula['points'],3)[:,0,:2]
    gaps=[]
    for a,b in zip(front[:-1],front[1:]):
        for t in np.linspace(0,1,101):
            x,y=a*(1-t)+b*t
            for c,d in zip(tongue[:-1],tongue[1:]):
                if min(c[0],d[0])<=x<=max(c[0],d[0]) and abs(d[0]-c[0])>1e-9:
                    gaps.append(y-c[1]-(x-c[0])/(d[0]-c[0])*(d[1]-c[1]))
    assert min(gaps)>=-2e-5 and min(gaps)<.002
    assert front[0]==pytest.approx(s['contours']['upper_cover'][9],abs=1e-5)
    # Contact releases again; there is no accumulated displacement/hysteresis.
    p[7]=0;assert e.snapshot(p)['uvula_contact_lift']==0


def test_extended_pose_survives_preset_file_sequence_and_animation(runtime):
    e=runtime.engine;p=e.presets['a'].copy();p[10]=6.8
    original={'id':'extended','name':'伸舌','params':p,'larynx_height':.65,'duration':.1}
    saved=runtime.invoke('presets/save',{'presets':[original]})
    assert runtime.invoke('presets/load')==saved
    frames=validate_frames(e,[original,dict(original,larynx_height=-.4)])
    doc=make_document(e,frames,[]);assert doc['version']==2
    assert sequence_document(e,json.loads(json.dumps(doc)))==doc
    downgraded=dict(doc,version=1,model='VTL-2.4-JD2')
    with pytest.raises(ValueError,match='版本 2'):sequence_document(e,downgraded)
    runtime.invoke('keyframes',{'frames':frames});assert runtime.invoke('keyframes/load')['frames']==frames
    mid=sample_frame(frames,.05);assert mid['larynx_height']==pytest.approx(.125)
    prepared=prepare_animation(e,frames)
    assert np.isfinite(prepared['audio']).all() and np.std(prepared['audio'])>1e-7
    last=json.loads(zlib.decompress(prepared['pictures'][-1]))
    assert last['larynx_height']==-.4
    np.testing.assert_array_equal(last['tube_areas'],e.snapshot(p,larynx_height=-.4)['tube_areas'])
    old=make_document(e,[{'params':e.presets['a']}],[])
    assert old['version']==1 and sequence_document(e,old)['frames'][0].get('larynx_height',0)==0


def test_runtime_pose_and_preview_share_the_extension(runtime):
    e=runtime.engine
    state=runtime.invoke('pose',{'params':e.presets['a'],'revision':1,'larynx_height':.3})
    assert runtime.live.larynx_height==state['larynx_height']==.3
    np.testing.assert_allclose(e.prepare_tube(runtime.live.params,runtime.live.lip_width,runtime.live.larynx_height)['areas'],state['tube_areas'],rtol=0,atol=5.1e-7)
