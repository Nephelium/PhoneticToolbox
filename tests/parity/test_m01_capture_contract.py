"""M01-A: independent old behavior, never v3-generated expectations."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('baseline_support_m01', ROOT / 'scripts/baseline_support.py')
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def manifest():
    return support.load_json(ROOT / 'tests/fixtures/m01/manifest.json')


def case(name):
    item = next(c for c in manifest()['cases'] if c['id'] == name)
    return support.load_json(ROOT / item['baseline_path'])['scientific']


def test_independent_producer_hashes_and_repeat_evidence():
    data = manifest()
    assert data['v3_algorithm_parity'] == 'not_implemented'
    for row in data['cases']:
        path = ROOT / row['baseline_path']
        assert support.sha(path) == row['sha256']
        result = support.load_json(path)
        assert result['producer']['root_kind'] == 'original_v2'
        assert result['producer']['modules_sha256']
        assert row['repeat_count'] == 2 and row['repeat_differences'] == []


def test_gui_service_and_parameter_subset_columns():
    full, gui = case('SERVICE-NONE'), case('GUI-ALL')
    assert len(full['columns']) == 79 and len(gui['columns']) == 77
    assert set(full['columns']) - set(gui['columns']) == {'SOE_pF0', 'SOE_rF0'}
    assert case('SERVICE-EMPTY')['columns'] == full['columns']
    for name, key in [('SELECT-PRAAT','pF0'), ('SELECT-REAPER','rF0'),
                      ('SELECT-INTENSITY','Intensity'), ('SELECT-ENERGY','Intensity')]:
        selected = case(name)
        assert selected['columns'] == ['Time_s', key]
        assert not support.compare(selected['tracks'][key], full['tracks'][key])
    assert case('GUI-CONTROLS')['empty_selection_rejected']
    assert case('GUI-CONTROLS')['cancel_preserves_selection']


def test_associations_keep_all_tiers_and_four_lip_parameters():
    result = case('ASSOCIATED')
    assert len(result['columns']) == 83
    assert result['columns'][-2:] == ['text_音节', 'text_IPA']
    assert result['tracks']['text_IPA']['values'][:40] == ['aː'] * 40
    assert result['tracks']['text_IPA']['values'][40:80] == ['iː'] * 40
    for key in ['LipArea','LipWidth','LipOpen','LipCirc']:
        assert key in result['columns']


def test_lip_anchor_offset_duplicates_and_nan_interpolation():
    result = case('LIP-TIME')
    expected = [None, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, None]
    for mode in ['metadata','companion','relative']:
        tracks = result['modes'][mode]
        for key, scale in [('LipArea',1),('LipWidth',2),('LipOpen',3),('LipCirc',4)]:
            assert tracks[key]['values'] == [None if x is None else x*scale for x in expected]
            assert tracks[key]['nonfinite'] == [1,0,0,0,0,0,0,0,1]
    assert result['insufficient'] == {}
    assert 'LipWidth' not in result['bad_length']


def test_every_setting_saved_and_reaches_its_actual_call():
    controls = case('GUI-CONTROLS')['settings']
    audit = support.load_json(ROOT / 'docs/modules/evidence/M01-parameter-settings.json')
    assert set(controls) == {s['key'] for s in audit['settings']}
    for row in audit['settings']:
        entry = controls[row['key']]
        assert entry['default'] == row['default']
        assert entry['range'] == row['range']
        result = case('SETTING-' + row['key'])
        assert result['config'][row['key']] == entry['changed']
        assert result['setting_observed'] is True
    destinations = {
        'silence_threshold':('compute_silence_mask','threshold_ratio'),
        'energy_window_ms':('compute_energy','energy_window_ms'),
        'frameshift_ms':('compute_praat_f0_track','frameshift_ms'),
        'windowsize_ms':('compute_jitter_shimmer','window_ms'),
        'smooth_win_size':('smooth_preserving_gaps','window_size'),
        'lip_smooth_win_size':('smooth_lip','lip_win'),
        'n_periods':('compute_spectral_features_batch','N_periods'),
        'num_formants':('compute_praat_formants','num_formants'),
        'max_formant':('compute_praat_formants','max_formant'),
        'min_f0':('compute_praat_f0_track','min_f0'),
        'max_f0':('compute_praat_f0_track','max_f0'),
        'reaper_hilbert':('compute_reaper_f0','hilbert'),
        'reaper_no_highpass':('compute_reaper_f0','no_highpass'),
    }
    for key,(function,argument) in destinations.items():
        result=case('SETTING-'+key)
        assert any(call[argument]==controls[key]['changed'] for call in result['observed_calls'][function])
    assert case('SETTING-only_voiced')['config']['only_voiced'] is False


def test_actual_backend_fallback_not_just_requested_label():
    native = case('SERVICE-NONE')
    assert any(p['returncode'] == 0 for p in native['native_processes'])
    fallback = case('REAPER-FAILURE')
    assert any(p['returncode'] != 0 for p in fallback['native_processes'])
    assert fallback['backend_returns']['run_python_impl']['finite_positive'] > 0
    wm = case('IRAPT-FAILURE')
    assert wm['fault_injection'] == 'irapt_raises'
    assert wm['backend_returns']['_extract_f0_for_wm']['finite_positive'] > 0
    assert wm['praat_wm_fallback_observed']


def test_exports_roundtrip_and_second_format_failure_boundary():
    result = case('ASSOCIATED')['export']
    assert result['xlsx_equal'] and result['sqlite_equal']
    assert result['sqlite_table'] == 'params'
    assert 'idx_params_time' in result['sqlite_indexes']
    assert result['display_columns'][1] == 'F0 - Praat'
    empty = case('EMPTY-EXPORT')['export']
    assert empty['status'] == 'error' and empty['xlsx_exists']
    assert empty['error_type'] == 'ValueError'
    formula = case('FORMULA-EXPORT')['export']
    assert formula['status'] == 'returned' and formula['formula_cells'] > 0
    assert not formula['xlsx_equal'] and formula['sqlite_equal']


def test_comparison_rejects_time_missing_backend_and_meaningful_value_changes():
    original = case('SERVICE-NONE')
    for edit in ['time','mask','value','backend']:
        changed = copy.deepcopy(original)
        if edit == 'time': changed['time_axis'][1] += 1e-12
        if edit == 'mask': changed['tracks']['pF0']['nonfinite'][0] = 3
        if edit == 'value':
            i = next(i for i,v in enumerate(changed['tracks']['pF0']['values']) if v is not None)
            changed['tracks']['pF0']['values'][i] += 100
        if edit == 'backend': changed['native_processes'][0]['returncode'] = 99
        assert support.compare(original, changed)


def test_environment_audit_does_not_treat_empty_downloads_as_installable_wheels():
    report=support.load_json(ROOT/'docs/testing/m01-environment-audit.json')
    packages=report['reference_packages']
    assert packages['numpy']['version']=='2.2.6'
    assert packages['scipy']['version']=='1.16.3'
    assert packages['praat-parselmouth']['version']=='0.4.7'
    assert all(p['matches_p03'] for p in packages.values() if p)
    assert report['not_installed_or_locked_this_turn']
    for wheel in report['wheel_candidates']:
        if wheel['size_bytes']==0: assert not wheel['valid_wheel_zip']


@pytest.mark.parametrize('problem',['partial','duplicate','existing'])
def test_freeze_refuses_incomplete_or_overwritten_oracles(tmp_path,monkeypatch,problem):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('capture_m01_test',ROOT/'scripts/capture_m01_baseline.py')
    capture=importlib.util.module_from_spec(spec); spec.loader.exec_module(capture)
    monkeypatch.setattr(capture,'ROOT',tmp_path)
    monkeypatch.setattr(capture,'OUTPUT',tmp_path/'output')
    folder=tmp_path/'output/run'; folder.mkdir(parents=True)
    rows=[{'id':c['id']} for c in capture.cases()]
    if problem=='partial': rows=rows[:1]
    if problem=='duplicate': rows.append(rows[0])
    support.write_json(folder/'capture-summary.json',{'cases':rows})
    protected=tmp_path/'tests/fixtures/m01'
    if problem=='existing':
        protected.mkdir(parents=True); (protected/'sentinel').write_bytes(b'original oracle')
    with pytest.raises(ValueError): capture.freeze(folder)
    if problem=='existing': assert (protected/'sentinel').read_bytes()==b'original oracle'
    else: assert not protected.exists()
