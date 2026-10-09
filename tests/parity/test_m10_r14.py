"""M10-R14 native tissue attachment, thickness, and dense sections."""
from pathlib import Path
import numpy as np
import pytest
from phonetic_core.vocal_tract.engine import Engine


def test_dense_sections_and_blade_material_station():
    e=Engine(resource_dir=Path(__file__).resolve().parents[2]/'resources/vocal_tract/native')
    try:
        p=e.presets['a'].copy();p[9]=-2.1;p[10]=1.5;p[11]=.25;p[12]=2.25;p[13]=-1.3
        s=e.snapshot(p)
        assert len(s['centerline'])==len(s['airway_sections'])==257
        assert s['section']==128
        assert e.snapshot(p,section=10000)['section']==256
        def anchor(s):
            r=s['tongue_blade_rib'];assert 0<r<32
            c=np.asarray(s['contours']['tongue']);a=int(r);f=r-a
            return c[a]*(1-f)+c[a+1]*f
        base=anchor(s)
        assert np.linalg.norm(base-p[12:14])>.3  # old off-surface construction point
        for idx,direction in [(12,-1),(13,-1),(13,1)]:
            q=p.copy();q[idx]+=.16*direction;t=e.snapshot(q);delta=anchor(t)-base
            assert .07<direction*delta[idx-12]<.15
            np.testing.assert_array_equal(s['meshes'][3]['vertices'],t['meshes'][3]['vertices'])
        tip=np.asarray(s['contours']['tongue'][:33])
        # A model-thickness acceptance bound, not an anatomical population claim.
        y=p[11];hits=[]
        for a,b in zip(tip,tip[1:]):
            if (a[1]<=y<b[1]) or (b[1]<=y<a[1]):
                hits.append(a[0]+(y-a[1])/(b[1]-a[1])*(b[0]-a[0]))
        hits.append(tip[-1,0])
        assert max(hits)-min(hits)>.5
        assert np.isfinite(s['tube_areas']).all()
    finally:e.close()
