"""Three-second IPA stop acceptance on the same native core used by all hosts."""
import copy
import os
from pathlib import Path

import numpy as np
import pytest

from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.sequence_controls import ipa_sequence
from phonetic_core.vocal_tract.stop_timing import validate_stop_timing
from phonetic_core.vocal_tract.trajectory import sample_trajectory
from phonetic_core.vocal_tract.animation import prepare_animation
from phonetic_core.vocal_tract.document import make_document, sequence_document

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(params=['JD2','W02'])
def engine(request):
    value=Engine(request.param,resource_dir=Path(os.environ.get('PTB_M10_TEST_RESOURCES',ROOT/'resources/vocal_tract/native')))
    yield value
    value.close()

@pytest.mark.parametrize('symbol',['b','d','ɡ','t'])
@pytest.mark.parametrize('duration',[.05,.2,2.008,2.1,3.])
def test_ipa_uniform_duration_roundtrips_without_shortening_or_changing_vot(engine,symbol,duration):
    catalog=[dict(name=n,params=list(engine.presets['a']),f0=engine.default_f0,lip_width=1.) for n in (symbol,'a')]
    original=copy.deepcopy(catalog)
    frames=ipa_sequence(engine,symbol+'a',catalog,duration,stop_timing=True)['frames']
    assert catalog==original
    assert [f['duration'] for f in frames]==[duration,duration]
    timing=frames[0]['stop_timing']
    assert sum(timing[k] for k in ('closure','release','aspiration'))==pytest.approx(duration,abs=1e-12)
    if symbol!='t':
        assert timing['aspiration']==0
        assert timing['closure']==duration-min(.008,duration*.1)
        assert timing['vot']==-min(.03,timing['closure'])
    restored=sequence_document(engine,make_document(engine,frames,[]))
    assert restored['frames']==frames

def test_three_second_voiced_stop_retains_closed_pose_negative_vot_and_exact_pcm_length(engine):
    meta=engine.metadata()
    catalog=[{**meta['consonants']['t'],'name':'b','f0':engine.default_f0,'lip_width':1.},dict(name='a',params=list(engine.presets['a']),f0=engine.default_f0,lip_width=1.)]
    frames=ipa_sequence(engine,'ba',catalog,3.,stop_timing=True)['frames']
    stop=frames[0]['stop_timing']
    assert stop==dict(closure=2.992,release=.008,aspiration=0.,vot=-.03)
    assert sample_trajectory(frames,2.95)['source']['vibration']==0
    assert sample_trajectory(frames,2.97)['source']['vibration']==1
    assert sample_trajectory(frames,2.97)['params']==frames[0]['params']
    rendered=prepare_animation(engine,frames,pictures_enabled=False)
    assert len(rendered['audio'])==6*engine.sr
    assert np.isfinite(rendered['audio']).all()

def test_stop_limits_keep_minimum_release_and_three_second_total():
    assert validate_stop_timing(dict(closure=2.999,release=.001,aspiration=0,vot=-2.999),3.)
    for timing,duration in [(dict(closure=3.,release=.001,aspiration=0,vot=0),3.001),(dict(closure=2.999,release=.002,aspiration=0,vot=0),3.001)]:
        with pytest.raises(ValueError):validate_stop_timing(timing,duration)
