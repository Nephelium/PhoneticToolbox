"""M15 prompt settings persist with the account; experimental records stay local."""
import pytest
from ptb_api.preferences import validate_values


def test_prompt_text_and_visibility_are_account_preferences():
    values = {'responsePrompt': '请判断 X，更接近哪一项？', 'showResponsePrompt': True}
    assert validate_values('M15', values) == values


def test_audition_gain_and_experiment_volume_are_distinct_account_preferences():
    assert validate_values('APP',{'automaticAudioGain':False})=={'automaticAudioGain':False}
    assert validate_values('M15',{'playbackVolume':.42})=={'playbackVolume':.42}
    for value in ('true',1,None):
        with pytest.raises(ValueError):validate_values('APP',{'automaticAudioGain':value})
    for value in (-.1,1.1,True,'0.5',None):
        with pytest.raises(ValueError):validate_values('M15',{'playbackVolume':value})


@pytest.mark.parametrize('field', ['attempts', 'answers', 'correctKey', 'trials', 'sessions'])
def test_perception_research_data_are_not_preferences(field):
    with pytest.raises(ValueError, match='invalid_settings_field'):
        validate_values('M15', {field: []})
