"""M04 original-V2 fixture comparison, without importing original code."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from phonetic_core.lpc import LPCConfig, compute_spectrum, mono_samples, extract_label, next_tier_name
from phonetic_core.models.associations import Interval, Tier

FIXTURE = Path(__file__).resolve().parents[1]/'fixtures/m04'
META = json.loads((FIXTURE/'result.json').read_text('utf-8'))


@pytest.fixture(scope='module')
def arrays():
    manifest = json.loads((FIXTURE/'manifest.json').read_text('utf-8'))
    for name, key in [('result.json', 'result_sha256'), ('arrays.npz', 'arrays_sha256')]:
        assert hashlib.sha256((FIXTURE/name).read_bytes()).hexdigest() == manifest[key]
    with np.load(FIXTURE/'arrays.npz', allow_pickle=False) as source:
        return dict(source)


def identical(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('name', list(META['cases']))
def test_original_spectrum(name, arrays):
    case = META['cases'][name]
    audio = arrays[name+'.input'].copy()
    before = audio.tobytes()
    if case['status'] == 'error':
        with pytest.raises(ValueError):
            compute_spectrum(audio, case['rate'], LPCConfig(**case['config']))
    else:
        result = compute_spectrum(audio, case['rate'], LPCConfig(**case['config']))
        identical(result.frequencies_hz, arrays[name+'.frequency'])
        identical(result.magnitude_db, arrays[name+'.db'])
        assert [result.amp_min_db, result.amp_max_db] == case['y_range']
    assert audio.tobytes() == before


@pytest.mark.parametrize('name', ['pcm16', 'pcm32', 'uint8', 'float32', 'stereo16'])
def test_original_sample_conversion(name, arrays):
    samples = arrays['wav.'+name+'.raw'].copy()
    before = samples.tobytes()
    identical(mono_samples(samples), arrays['wav.'+name+'.mono'])
    assert samples.tobytes() == before


@pytest.fixture
def tiers():
    return (Tier('phones', (Interval(0., .2, 'ɑ̃˥'), Interval(.2, .4, 'b'),
        Interval(.4, .6, ''), Interval(.6, .8, 'b'), Interval(.8, 1., '末'))),
        Tier('words', (Interval(0., 1., '词'),)))


@pytest.mark.parametrize('key', list(META['labels']))
def test_original_labels(key, tiers):
    a, b = map(float, key.split(':'))
    assert extract_label(tiers, 'phones', a, b) == META['labels'][key]


def test_original_tier_cycle(tiers):
    assert [next_tier_name(tiers, x) for x in [None, 'phones', 'words', 'missing']] == META['tiers']
