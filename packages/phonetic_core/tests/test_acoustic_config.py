import dataclasses
import pytest
from phonetic_core.models.acoustic import AcousticConfig
from phonetic_core.acoustic.catalog import PARAMETER_MAPPING


def test_config_snapshot_and_legacy_defaults():
    selection = ['pF0']
    config = AcousticConfig(selected_parameter_keys=selection)
    selection.append('rF0')
    assert config.selected_parameter_keys == ('pF0',)
    assert config.frameshift_ms == 5.0 and config.windowsize_ms == 40.0
    assert len(PARAMETER_MAPPING) == 80
    assert 'reaper_bin_path' not in dataclasses.asdict(config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.frameshift_ms = 10.


@pytest.mark.parametrize('values', [dict(frameshift_ms=0), dict(min_f0=900),
    dict(max_formant=0), dict(silence_threshold=float('nan')), dict(smooth_win_size=-1)])
def test_invalid_config_rejected(values):
    with pytest.raises(ValueError):
        AcousticConfig(**values)
