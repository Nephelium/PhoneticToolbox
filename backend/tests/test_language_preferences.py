"""UI language is a bounded account setting, separate from research inputs."""
import pytest
from ptb_api.preferences import validate_values


@pytest.mark.parametrize('code',['zh-CN','en','ja'])
def test_supported_languages_are_accepted_without_changing_other_preferences(code):
    value={'uiLocale':code,'automaticAudioGain':False,'pageScale':125}
    assert validate_values('APP',value)==value


@pytest.mark.parametrize('value',[None,True,0,[],{},'','EN','zh','../ja','<script>','data:en'])
def test_unregistered_or_malformed_language_is_rejected(value):
    with pytest.raises(ValueError,match='invalid_settings_field'):
        validate_values('APP',{'uiLocale':value})


def test_language_is_not_accepted_as_a_module_scientific_parameter():
    with pytest.raises(ValueError,match='invalid_settings_field'):
        validate_values('M04',{'uiLocale':'en'})
