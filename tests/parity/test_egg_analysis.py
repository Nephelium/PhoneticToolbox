"""M03-B parity against independently captured original-v2 arrays."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from phonetic_core.egg import EGGConfig, prepare, analyze_events, cq_segment, events_segment
from phonetic_core.egg.f0 import praat_pitch, gci_pitch, glottal_movement
from phonetic_core.egg.inverse import inverse_filter

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures/m03'


@pytest.fixture(scope='module')
def reference():
    meta = json.loads((FIXTURES/'EGG-SYN-PCM16.json').read_text('utf-8'))
    with np.load(FIXTURES/'EGG-SYN-PCM16.npz', allow_pickle=False) as values:
        return meta, dict(values)


def equal(actual, expected):
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


def unpack(value, arrays):
    if isinstance(value, dict) and 'array' in value:
        return arrays[value['array']]
    if isinstance(value, list):
        return [unpack(v, arrays) for v in value]
    return value


@pytest.mark.parametrize('index', range(8))
def test_methods_thresholds_global_and_both_roi_policies(reference, index):
    meta, arrays = reference
    cfg = EGGConfig(**meta['variants'][index]['config'])
    # Reconstruct service input from the frozen normalized channels, preserving dtype.
    samples = np.column_stack((arrays['load.egg_signal_raw'], arrays['load.audio_signal']))
    state = prepare(samples, 44100, cfg)
    for name in ('time_vector', 'egg_signal_raw', 'audio_signal', 'egg_signal_processed'):
        equal(getattr(state, name), arrays['load.'+name])
    result = analyze_events(state, cfg)
    expected = meta['variants'][index]
    assert [result.gci_times, result.goi_times, result.peak_times] == expected['events']
    for actual, old in zip((result.gci_f0_times, result.gci_f0_values), expected['gci_f0']):
        equal(actual, unpack(old, arrays))
    for label, modes in expected['roi'].items():
        for mode, item in modes.items():
            start, end = item['bounds_s']
            for fn, kind in ((cq_segment, 'cq'), (events_segment, 'events')):
                actual = fn(result, start, end, cfg, use_raw_signal=mode=='raw')
                old = unpack(item[kind]['value'], arrays)
                for a, b in zip(actual, old):
                    equal(a, b)


def test_praat_array_input_matches_actual_values_and_legacy_time(reference):
    _, arrays = reference
    track = praat_pitch(arrays['load.audio_signal'], 44100)
    equal(track.values, arrays['praat.legacy.values'])
    equal(track.times, arrays['praat.actual.times'])
    equal(track.legacy_times, arrays['praat.legacy.times'])


@pytest.mark.parametrize('order,key', [(None,'auto'), (12,'explicit')])
def test_inverse_array_output(reference, order, key):
    meta, arrays = reference
    variant = next(v for v in meta['variants'] if v['config']['gci_method']=='slope'
                   and v['config']['goi_method']=='scale' and v['config']['auto_prominence'])
    audio = arrays['load.audio_signal'][:int(.12*44100)]
    events = np.array([t for t in variant['events'][0] if t < len(audio)/44100])
    old_audio = audio.copy()
    actual = inverse_filter(audio, 44100, events, lp_order=order)
    equal(actual, arrays['inverse.'+key])
    equal(audio, old_audio)


@pytest.mark.parametrize('label,times', [('low',[0,.02,.04,.06]),
                                       ('outlier',[0,.005,.010,.03,.035,.04]),
                                       ('duplicate',[0,.01,.01,.02])])
def test_gci_pitch_legacy_masks(reference, label, times):
    meta, arrays = reference
    result = gci_pitch(times)
    for actual, old in zip(result, meta['analytic']['gci_f0'][label]):
        equal(actual, unpack(old, arrays))


def test_glottal_heuristic(reference):
    meta, _ = reference
    result = glottal_movement(np.array([0,.01,.02,.03,.14,.15,.16]),
                             np.array([100,130,100,140,400,100,np.nan]))
    assert [list(event) for event in result] == meta['analytic']['movement']
