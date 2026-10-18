"""M10 settings whitelist, without a database or identity fixture."""
import pytest
from ptb_api.preferences import validate_values


@pytest.mark.parametrize('setting',[None,dict(jitter=10,shimmer=0,seed=1),dict(jitter=0,shimmer=100,seed=4294967295)])
def test_source_settings_are_account_preferences(setting):
    values={'frame':{'perturbation':setting}}
    assert validate_values('M10',values)==values


@pytest.mark.parametrize('setting',[dict(jitter=101),dict(shimmer=-1),dict(seed=True),dict(seed=4294967296),dict(audio_base64='unwanted')])
def test_source_settings_reject_invalid_controls_and_content(setting):
    with pytest.raises(ValueError):validate_values('M10',{'frame':{'perturbation':setting}})
