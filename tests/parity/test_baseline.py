import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('support', ROOT / 'scripts/baseline_support.py')
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def test_comparator_rejects_changed_time_mask_shape_and_large_numeric_error():
    expected = {'time_axis': [0.0, 0.005], 'values': [120.0, None], 'nonfinite': [0, 1]}
    assert not support.compare(expected, expected)
    for changed in [dict(expected, time_axis=[0.0, 0.005000000001]),
                    dict(expected, nonfinite=[0, 2]), dict(expected, values=[120.0]),
                    dict(expected, values=[125.0, None]), dict(expected, values=[120.0, 0.0])]:
        assert support.compare(expected, changed)
    assert support.compare({'config': {'frameshift_ms': 5.0}}, {'config': {'frameshift_ms': 5.000000001}})
    assert support.compare({'cq_sq_roi': [[0.5]]}, {'cq_sq_roi': [[0.500000000001]]})


def test_fixture_headers_match_recipes(tmp_path):
    import wave
    for recipe in support.RECIPES:
        path = support.create_fixture(tmp_path, recipe)
        if recipe['kind'] == 'broken':
            with pytest.raises(wave.Error):
                wave.open(str(path))
        else:
            with wave.open(str(path)) as source:
                assert source.getnframes() == recipe['frames']
                assert source.getframerate() == recipe['rate']
                assert source.getnchannels() == recipe['channels']


@pytest.mark.parametrize('key', ['time_axis', 'time_vector', 'rTimes', 'gci_times', 'gci_f0_times'])
def test_all_timestamps_are_exact(key):
    assert support.compare({key: [0.005]}, {key: [0.005000000001]})
    assert not support.compare({'frequency': [120.0]}, {'frequency': [120.00000001]})


def test_frozen_baseline_has_independent_producer_and_all_cases():
    manifest = support.load_json(ROOT / 'tests/fixtures/manifest.json')
    assert manifest['producer']['kind'] == 'original-v2-conda-process'
    assert manifest['v3_algorithm_parity'] == 'not_implemented'
    assert {item['id'] for item in manifest['public_cases']} == {r['id'] for r in support.RECIPES}
    for item in manifest['public_cases']:
        path = ROOT / item['baseline_path']
        assert support.sha(path) == item['baseline_sha256']
        baseline = support.load_json(path)
        assert baseline['case_id'] == item['id']
        assert baseline['producer']['root_kind'] == 'original_v2'
        assert baseline['producer']['modules_sha256']
        assert not support.compare(baseline['scientific'], baseline['scientific'])


def test_public_fixture_bytes_match_capture(tmp_path):
    manifest = support.load_json(ROOT / 'tests/fixtures/manifest.json')
    for item in manifest['public_cases']:
        recipe = next(r for r in support.RECIPES if r['id'] == item['id'])
        assert support.sha(support.create_fixture(tmp_path, recipe)) == item['input_sha256']


def test_analytic_time_grid_and_legacy_boundary_outcomes():
    def scientific(case):
        return support.load_json(ROOT / 'tests/fixtures/golden' / (case + '.json.gz'))['scientific']
    vowel = scientific('SYN-VOWEL-44100')
    assert vowel['audio']['sample_rate_hz'] == 44100
    assert vowel['result']['time_axis']['values'] == [i * 5 / 1000 for i in range(160)]
    # Known legacy metadata defect; changing it requires a separately reviewed behavior fix.
    assert vowel['result']['sampling_rate'] == 16000
    silence = scientific('SYN-SILENCE-16000')['result']
    assert all(v is None for v in silence['f0_praat']['values'])
    assert scientific('SYN-BROKEN')['status'] == 'error'
    for case in ['SYN-EMPTY', 'SYN-ONE-FRAME']:
        result = scientific(case)
        assert result['status'] == 'returned'
        assert result['result']['time_axis']['values'] == []
    egg = scientific('SYN-EGG-44100')
    assert egg['channel_selection']['flip_channels'] is False
    assert egg['result']['time_vector']['first'] == 0.0
    assert egg['result']['time_vector']['last'] == (35280 - 1) / 44100
    assert egg['result']['gci_times']


def test_parameter_coverage_and_confirmed_private_stereo_mapping():
    contract = support.load_json(ROOT / 'tests/fixtures/parameter-contract.json')
    assert contract['declared_parameter_count'] == 80
    assert contract['captured_parameter_count'] == 78
    assert len({row['key'] for row in contract['parameters']}) == 82
    missing = {row['key'] for row in contract['parameters'] if row['coverage'] != 'captured'}
    assert missing == {'LipArea', 'LipWidth', 'LipOpen', 'LipCirc'}
    manifest = support.load_json(ROOT / 'tests/fixtures/manifest.json')
    cases = {item['id']: item for item in manifest['private_cases']}
    for case, flip in [('EGG-01', False), ('EGG-05', True)]:
        assert cases[case]['audio']['channels'] == 2
        assert cases[case]['channel_selection']['flip_channels'] is flip
