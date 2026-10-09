"""Regression for the reported spike at the raised, retracted tongue tip."""
from pathlib import Path
import numpy as np
from phonetic_core.vocal_tract.engine import Engine


def test_retroflex_tip_has_a_resolved_round_cap_and_finite_tube():
    root=Path(__file__).resolve().parents[2]
    engine=Engine(resource_dir=root/'resources/vocal_tract/native')
    try:
        p=engine.presets['a'].copy()
        p[9]=-2.1;p[10]=1.5;p[11]=.25;p[12]=2.6;p[13]=-1.3
        state=engine.snapshot(p)
        points=np.array(state['contours']['tongue'][:34])
        segments=np.diff(points,axis=0)
        segments/=np.linalg.norm(segments,axis=1)[:,None]
        turns=np.degrees(np.arccos(np.clip(np.sum(segments[:-1]*segments[1:],axis=1),-1,1)))
        cap=points[1:-1,1]>.2
        # R12 had three cap samples and a 139-degree reversal at its apex.
        assert np.count_nonzero(points[:,1]>.2)>=7
        assert np.max(turns[cap])<40
        assert np.isfinite(state['tube_areas']).all()
        assert np.isfinite(state['transfer_db']).all()
    finally:
        engine.close()
