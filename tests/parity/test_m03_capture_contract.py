"""M03-A: independent v2 evidence, not a v3 algorithm accuracy claim."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'tests/fixtures/m03'


@pytest.fixture(scope='module')
def manifest():
    return json.loads((FIXTURES / 'manifest.json').read_text('utf-8'))


def test_provenance_and_coverage(manifest):
    assert manifest['producer'] == 'original-v2-conda-process'
    assert manifest['repeat_comparison'] == 'exact_arrays_masks_and_metadata'
    assert len(manifest['public_cases']) >= 9
    assert {c['id'] for c in manifest['private_cases']} == {'EGG-01', 'EGG-05'}
    assert all(c['repeat_equal'] and c['input_unchanged'] for c in
               manifest['public_cases'] + manifest['private_cases'])
    assert manifest['source_unchanged']


def evidence(manifest, case_id):
    case = next(c for c in manifest['public_cases'] if c['id'] == case_id)
    path = FIXTURES / case['metadata']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case['metadata_sha256']
    data = json.loads(path.read_text('utf-8'))
    arrays_path = FIXTURES / case['arrays']
    assert hashlib.sha256(arrays_path.read_bytes()).hexdigest() == case['arrays_sha256']
    arrays = np.load(arrays_path, allow_pickle=False)
    for key, description in data['arrays'].items():
        value = arrays[key]
        assert list(value.shape) == description['shape']
        assert str(value.dtype) == description['dtype']
        assert hashlib.sha256(value.tobytes()).hexdigest() == description['sha256']
        assert int(np.isnan(value).sum()) == description['nan_count']
    return data, arrays


@pytest.mark.parametrize('case_id', [
    'EGG-SYN-PCM16', 'EGG-SYN-FLOAT', 'EGG-SYN-SWAPPED', 'EGG-SYN-SILENCE',
    'EGG-SYN-MONO', 'EGG-SYN-SHORT', 'EGG-SYN-EMPTY', 'EGG-SYN-BROKEN',
    'EGG-SYN-MULTI',
])
def test_frozen_arrays_and_expected_input_outcomes(manifest, case_id):
    data, arrays = evidence(manifest, case_id)
    if case_id in {'EGG-SYN-MONO', 'EGG-SYN-MULTI', 'EGG-SYN-EMPTY', 'EGG-SYN-BROKEN'}:
        assert data['load']['status'] == 'error'
    else:
        assert data['load']['status'] == 'returned'
        assert arrays['load.time_vector'][0] == 0
    assert data['modules_sha256']['phonetic_toolbox.services.egg_service']


def test_eight_method_threshold_combinations_and_roi_policies(manifest):
    data, arrays = evidence(manifest, 'EGG-SYN-PCM16')
    assert len(data['variants']) == 8
    for variant in data['variants']:
        assert variant['config']['gci_method'] in ('slope', 'scale')
        assert variant['config']['goi_method'] in ('slope', 'scale')
        for roi in ('middle', 'start', 'end', 'outside'):
            for mode in ('raw', 'filtered'):
                assert variant['roi'][roi][mode]['cq']['status'] == 'returned'
                assert variant['roi'][roi][mode]['events']['status'] == 'returned'
    assert any(k.endswith('.cq.1') and np.isfinite(v).any() for k, v in arrays.items())


def test_analytic_metrics_and_real_praat_frame_times(manifest):
    data, arrays = evidence(manifest, 'EGG-SYN-PCM16')
    np.testing.assert_allclose(arrays['analytic.cq.1'], [.6, .6], rtol=0, atol=2e-16)
    np.testing.assert_allclose(arrays['analytic.cq.2'], [1/3, -1/3], rtol=0, atol=3e-16)
    np.testing.assert_array_equal(arrays['praat.actual.values'], arrays['praat.legacy.values'])
    assert np.max(np.abs(arrays['praat.actual.times'] - arrays['praat.legacy.times'])) > .01
    assert data['inverse']['auto']['status'] == 'returned'
    assert data['inverse']['no_gci']['value'] is None
    assert arrays['inverse.auto'].size > 0
    assert data['cancellation']['load'] is None
    assert data['cancellation']['events'] == [[], [], []]


def test_layout_defaults_and_actual_export_artifacts(manifest):
    data, arrays = evidence(manifest, 'EGG-SYN-PCM16')
    ui = data['ui']
    assert data['service_defaults']['goi_method'] == 'slope'
    assert ui['config']['gci_method'] == 'slope'
    assert ui['config']['goi_method'] == 'scale'
    positions = ui['geometry']
    assert positions['cq_canvas'][0] < positions['audio_zoom_canvas'][0]
    assert positions['cq_canvas'][1] < positions['spec_canvas'][1]
    assert positions['audio_zoom_canvas'][1] < positions['egg_zoom_canvas'][1]
    assert positions['timeline_canvas'][1] > positions['egg_zoom_canvas'][1]
    assert ui['timeline_window_s'] == 60
    assert ui['zoom_window_ms'] == 50
    assert len(data['exports']['single']['png']) == 3
    assert len(data['exports']['batch']['png']) == 3
    assert all(p['size'] == [1500, 900] for p in data['exports']['single']['png'])
    assert set(data['exports']['single']['columns']) >= {'Time (s)', 'CQ', 'SQ', 'F0_Praat (Hz)', 'F0_GCI (Hz)'}
    assert data['exports']['cancelled']['files'] == []
    assert data['exports']['errors']['failure_count'] == 1
    assert data['exports']['errors']['csv_count'] == 1
    assert np.isnan(arrays['export.masked.csv'][:, 1:]).all()
    assert np.isfinite(arrays['export.batch.csv'][:, 1:]).any()


def test_swapped_channels_restore_same_normalized_signals(manifest):
    _, normal = evidence(manifest, 'EGG-SYN-PCM16')
    _, swapped = evidence(manifest, 'EGG-SYN-SWAPPED')
    for role in ('egg_signal_raw', 'audio_signal', 'egg_signal_processed'):
        np.testing.assert_array_equal(normal['load.' + role], swapped['load.' + role])


def test_silence_is_empty_events_with_missing_pitch(manifest):
    data, arrays = evidence(manifest, 'EGG-SYN-SILENCE')
    assert data['variants'][0]['events'] == [[], [], []]
    assert np.isnan(arrays['praat.legacy.values']).all()


def test_metric_boundary_masks_and_heuristic_movement(manifest):
    data, arrays = evidence(manifest, 'EGG-SYN-PCM16')
    assert np.isnan(arrays['analytic.boundaries.1']).all()
    # v2 masks CQ independently: invalid CQ does not force its SQ to NaN.
    np.testing.assert_allclose(arrays['analytic.boundaries.2'][:2], [.2, .35/.95], rtol=0, atol=2e-16)
    assert np.isnan(arrays['analytic.multiple_peaks.2']).all()
    # Exact IEEE reciprocals of the declared event intervals, rather than rounded 50 Hz.
    np.testing.assert_array_equal(arrays['analytic.f0.low.1'], [1/.02, 1/(.04-.02), 1/(.06-.04)])
    assert data['analytic']['movement'] == [[0, 'Rise'], [.01, 'Fall'], [.14, 'Fall']]


def test_ignored_legacy_parameters_and_short_filter_warning(manifest):
    data, _ = evidence(manifest, 'EGG-SYN-PCM16')
    ignored = data['ignored_parameters']
    assert ignored['0.25_50'] == ignored['0.75_50'] == ignored['0.25_200']
    short, _ = evidence(manifest, 'EGG-SYN-SHORT')
    assert len(short['load']['warnings']) == 2
    assert all('Returning unfiltered data' in warning for warning in short['load']['warnings'])


def test_window_lengths_and_independent_f0_export_choices(manifest):
    data, arrays = evidence(manifest, 'EGG-SYN-PCM16')
    for window in (5, 20, 50):
        spectrum = data['spectrogram'][str(window)]
        assert len(spectrum) == 1
        assert arrays[spectrum[0]['data']['array']].shape[0] == int(44100*window/1000)//2+1
        assert spectrum[0]['clim'] == [-70, -10]
    assert data['exports']['no_praat']['columns'] == ['Time (s)', 'CQ', 'SQ', 'F0_GCI (Hz)']
    assert data['exports']['no_gci_f0']['columns'] == ['Time (s)', 'CQ', 'SQ', 'F0_Praat (Hz)']
    assert data['exports']['no_f0']['columns'] == ['Time (s)', 'CQ', 'SQ']


def test_legacy_padded_rows_are_explicitly_preserved(manifest):
    _, arrays = evidence(manifest, 'EGG-SYN-PCM16')
    times = arrays['export.single.csv'][:, 0]
    assert times.min() < .1 and times.max() > .5
    np.testing.assert_allclose(arrays['praat.actual.times'] - arrays['praat.legacy.times'],
                               .015, rtol=0, atol=1e-16)
