import hashlib
import json
import gzip
from pathlib import Path
from importlib.resources import files
import numpy as np
import pytest
from phonetic_core.acoustic.lip import interpolate_lip
from phonetic_core.acoustic.annotations import align_annotations
from phonetic_core.acoustic.f0_irapt import _sinc_hash_table
from phonetic_core.models.associations import Tier, Interval

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('mode', ['metadata','companion','relative'])
def test_lip_time_modes_against_independent_old_capture(mode):
    old=json.loads(gzip.decompress((ROOT/'tests/fixtures/m01/LIP-TIME.json.gz').read_bytes()))['scientific']
    times=[.5,0.,.25,.25,.75,np.nan]
    data={'metadata':{'lip_manual_offset':.125},'relative_times':[x+10 for x in times],
          'absolute_timestamps':[x+100 for x in times]}
    for i,key in enumerate(['area','outer_width','open','circularity'],1):
        data[key]=[3*i,1*i,2*i,99*i,4*i,5*i]
    data['area'][2]=np.nan
    if mode=='metadata': data['metadata']['audio_first_frame_time']=100.
    result=interpolate_lip(data,np.array(old['target_times']),companion_start=100. if mode=='companion' else None)
    for key,value in result.items():
        expected=np.array(old['modes'][mode][key]['values'],dtype=float)
        np.testing.assert_array_equal(value,expected)


def test_annotations_ceil_half_open_order_and_literal_text():
    tiers=(Tier('IPA',(Interval(.001,.010,'aː'),Interval(.010,.015,'=literal'))),)
    result=align_annotations(tiers,4,5.)
    assert result['text_IPA'].tolist()==['','aː','=literal','']
    duplicate=tiers+(Tier('IPA',(Interval(0.,.02,'替换'),)),)
    assert align_annotations(duplicate,4,5.)['text_IPA'].tolist()==['替换']*4


def test_sinc_resource_packaged_exactly_and_used():
    resource=files('phonetic_core.acoustic').joinpath('data/Sinc_hash_1000.mat')
    assert hashlib.sha256(resource.read_bytes()).hexdigest()=='e3e2fb01d67f722b7f13c1c9a0f4d62559a1ece77167343dfa42de7203f4d860'
    from scipy.io import loadmat
    with resource.open('rb') as f: expected=loadmat(f)['Sinc_hash']
    np.testing.assert_array_equal(_sinc_hash_table(),expected)
