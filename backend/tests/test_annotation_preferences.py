"""M12 link/snap controls are account preferences, annotation contents stay local."""
import pytest
from ptb_api.preferences import validate_values

def test_annotation_link_preferences():
    values = dict(snapEnabled=True, linkWordPhone=False, linkWordExtra=True, linkPhoneExtra=False)
    assert validate_values('M12', values) == values

@pytest.mark.parametrize('field', ['tiers', 'textgrid', 'selected', 'keyboardText'])
def test_annotation_contents_cannot_be_saved_as_preferences(field):
    with pytest.raises(ValueError, match='invalid_settings_field'):
        validate_values('M12', {field: []})
