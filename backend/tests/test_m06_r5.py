"""R5 retirement is explicit and old result manifests remain readable."""
from copy import deepcopy
import pytest
from ptb_api.m06_models import M06Request
from phonetic_core.synthesis.klatt.api import defaults,validate


def test_klatt_r4_snapshot_nonmutating_compatibility():
    old=defaults();old['render']=dict(method='klatt',pitch='original',spectral_ratio=1,aperiodicity_ratio=1,source_sha256=None)
    before=deepcopy(old)
    assert validate(old)==defaults() and old==before


@pytest.mark.parametrize('method',['world','psola'])
def test_retired_method_cannot_silently_use_klatt(method):
    old=defaults();old['render']=dict(method=method)
    with pytest.raises(ValueError,match='m06_removed_synthesis_method'):validate(old)


def test_harvest_and_resynthesis_action_are_unavailable():
    old=defaults();old['f0_method']='harvest'
    with pytest.raises(ValueError,match='m06_removed_f0_method'):validate(old)
    assert 'resynthesize' not in M06Request.model_json_schema()['properties']['action']['enum']
