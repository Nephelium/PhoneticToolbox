"""Independent recorded V2 name codec vectors; no MFA environment required."""
import pytest


@pytest.mark.parametrize('source,expected', [
    ('中文 1.wav', 'ptbx_e4b8ade696872031.wav'),
    ('a.b.wav', 'ptbx_61.b.wav'),
    ('Speaker 1', 'ptbx_537065616b65722031'),
    ('no_suffix', 'ptbx_6e6f5f737566666978'),
])
def test_v2_filename_vectors(source, expected):
    from phonetic_core.transcription.mfa_name_codec import encode_fs_name, decode_fs_name
    assert encode_fs_name(source) == expected
    assert decode_fs_name(expected) == source


def test_v2_defaults_and_retry_normalization():
    from ptb_api.m11_models import M11Config
    assert M11Config().model_dump() == {'beam': 10, 'retry_beam': 40}
    assert M11Config(beam=20, retry_beam=20).retry_beam == 80
    # Service behaviour: retry > beam is preserved even below 4 * beam.
    assert M11Config(beam=20, retry_beam=30).retry_beam == 30
