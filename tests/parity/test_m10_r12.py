"""M10-R12 real native blade depression, tip retraction, rigid dental poses."""
from pathlib import Path
import numpy as np
import pytest
from phonetic_core.vocal_tract.engine import Engine

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def engine():
    e=Engine(resource_dir=ROOT/'resources/vocal_tract/native')
    yield e
    e.close()


def dental(s):
    return [m['vertices'] for m in s['meshes'] if m['name'].endswith('_teeth')]


def test_blade_depresses_continuously_without_moving_teeth(engine):
    e=engine;p=e.presets['a'].copy();base=e.snapshot(p);last=None
    for y in np.linspace(p[13],-2.5,21):
        p[13]=float(y);s=e.snapshot(p)
        assert s['limited'][13]==pytest.approx(y,abs=1e-8)
        np.testing.assert_array_equal(dental(s),dental(base))
        assert s['limited'][8:12]==pytest.approx(base['limited'][8:12],abs=1e-8)
        tongue=np.array(s['contours']['tongue'][:34])
        if last is not None:assert np.max(np.linalg.norm(tongue-last,axis=1))<.12
        last=tongue
    assert not np.allclose(s['tube_areas'],base['tube_areas'])
    tube=e.prepare_tube(p)
    np.testing.assert_allclose(tube['areas'],s['tube_areas'],atol=5.1e-7,rtol=0)
    assert np.isfinite(s['transfer_db']).all()


def test_raised_tip_retracts_without_translating_body(engine):
    e=engine;p=e.presets['a'].copy();p[9]=-2.1;p[11]=.25;p[12]=2.6;p[13]=-1.3
    reference=None;last=None
    for x in np.linspace(3.0,1.5,31):
        p[10]=float(x);s=e.snapshot(p)
        assert s['limited'][10]==pytest.approx(x,abs=1e-8)
        assert s['limited'][8:10]==pytest.approx(p[8:10],abs=1e-8)
        if reference is None:reference=dental(s)
        np.testing.assert_array_equal(dental(s),reference)
        tongue=np.array(s['contours']['tongue'][:34])
        if last is not None:assert np.max(np.linalg.norm(tongue-last,axis=1))<.15
        last=tongue
        assert np.isfinite(s['transfer_db']).all()


def test_lower_teeth_follow_jaw_as_rigid_body(engine):
    e=engine;p=e.presets['a'].copy();base=e.snapshot(p)
    a=np.asarray(dental(base)[1]).reshape(-1,3)[::7]
    distances=np.linalg.norm(a[:,None]-a[None,:],axis=2)
    for angle in [-15,-8,0]:
        p[3]=angle;s=e.snapshot(p);b=np.asarray(dental(s)[1]).reshape(-1,3)[::7]
        np.testing.assert_array_equal(dental(s)[0],dental(base)[0])
        np.testing.assert_allclose(np.linalg.norm(b[:,None]-b[None,:],axis=2),distances,atol=3e-5,rtol=0)
        assert not np.array_equal(b,a)
