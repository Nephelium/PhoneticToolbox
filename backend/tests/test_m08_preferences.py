from ptb_api.preferences import validate_values

def test_multi_variant_preferences_do_not_require_database_migration():
    values=dict(speed='0.8, 1, 1.2',pitch_ratio='1, 1.25',pitch_hz='-20, 0, 20',pitchMode='hz')
    assert validate_values('M08',values)==values


def test_time_reference_settings_are_allowed_but_export_consent_is_not_persisted():
    import pytest
    values=dict(refs=[200],timeRefs=[0,0.125,1.5],timeRefInput='0.125',refDirection='time')
    assert validate_values('M08',values)==values
    with pytest.raises(ValueError,match='invalid_settings_field'):
        validate_values('M08',dict(numberExports=True))
