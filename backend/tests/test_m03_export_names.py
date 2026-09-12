from ptb_worker.egg_export_names import export_names


def test_source_and_actual_roi_names_match_v2_and_keep_internal_keys():
    assert export_names('元音 ɑ̃˥.wav','single',.1,.5,['egg_DATA.csv']) == {'egg_DATA.csv':'元音 ɑ̃˥_0_10s_0_50s_DATA.csv'}
    assert export_names('元音.wav','batch',0,.8,['egg_DATA.csv']) == {'egg_DATA.csv':'元音_DATA.csv'}
    assert export_names('元音.wav','inverse',0,.12,['egg_IF.wav']) == {'egg_IF.wav':'元音_0_00s_0_12s_IF.wav'}


def test_unsafe_or_long_names_cannot_escape_destination_or_collide_after_truncation():
    names=export_names('../CON:evil.wav','single',0,.5,['egg.ptb.json','egg_DATA.csv'])
    assert all(not any(c in value for c in '/\\:') and not value.startswith('.') for value in names.values())
    a=export_names('音'*300+'a.wav','batch',0,1,['egg_DATA.csv'])['egg_DATA.csv']
    b=export_names('音'*300+'b.wav','batch',0,1,['egg_DATA.csv'])['egg_DATA.csv']
    assert a!=b and max(len(a),len(b))<=220
