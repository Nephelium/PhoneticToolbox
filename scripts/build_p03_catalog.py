"""Describe each captured parameter without creating numerical expected results."""
import json
from pathlib import Path
from baseline_support import load_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    declared = load_json(ROOT / 'docs/modules/all-80-acoustic-parameters.json')
    scientific = load_json(ROOT / 'tests/fixtures/golden/SYN-VOWEL-44100.json.gz')['scientific']
    captured = set(scientific['dataframe_columns']) - {'Time_s'}
    rows = []
    for item in declared + [{'key': 'SOE_pF0'}, {'key': 'SOE_rF0'}]:
        key = item['key']
        if key.startswith('Lip'):
            unit, source = 'unknown_without_input_metadata', 'services/io/lip.py'
        elif key in {'pF0', 'rF0', 'pF1', 'pF2', 'pF3', 'pF4', 'pB1', 'pB2', 'pB3', 'pB4'}:
            unit, source = 'Hz', 'services/acoustic_service.py'
        elif key.startswith(('Jitter_', 'Shimmer_')):
            unit, source = 'percent', 'core/acoustic/jitter_shimmer.py'
        elif key.startswith('SpectralSlope_'):
            unit, source = 'dB_per_decade', 'core/acoustic/spectral_slope.py'
        elif key.startswith('SHR_'):
            unit, source = 'amplitude_ratio', 'core/acoustic/shr.py'
        elif key.startswith('SOE_'):
            unit, source = 'normalized_ZFF_difference', 'core/acoustic/soe.py'
        elif key == 'Intensity':
            unit, source = 'dB_legacy_amplitude_reference_not_measured_SPL', 'core/acoustic/energy.py'
        elif key.startswith('CPP_'):
            unit, source = 'dB', 'core/acoustic/cpp.py'
        elif key.startswith('HNR'):
            unit, source = 'dB', 'core/acoustic/hnr.py'
        else:
            unit, source = 'dB_uncalibrated_magnitude_or_difference', 'services/acoustic_service.py'
        assert (ROOT / 'phonetic_toolbox' / source).is_file(), source
        rows.append({'key': key, 'unit_in_legacy_code': unit, 'source': 'phonetic_toolbox/' + source,
                     'coverage': 'captured' if key in captured else 'missing_lip_input',
                     'rtol': 1e-7, 'atol': 1e-10, 'time_and_missing_masks': 'exact',
                     'nonfinite_reason': 'legacy_unspecified_unless_separate_mask_evidence',
                     'tolerance_scope': 'same-platform repeat capture; scientific accuracy not inferred'})
    assert captured == {row['key'] for row in rows if row['coverage'] == 'captured'}
    result = {'task': 'P03', 'declared_parameter_count': len(declared),
              'captured_parameter_count': len(captured), 'parameters': rows,
              'egg': {'event_times_unit': 's', 'event_times_comparison': 'exact',
                      'f0_unit': 'Hz', 'CQ_SQ_unit': 'ratio', 'rtol': 1e-7, 'atol': 1e-10,
                      'full_preprocessing_arrays': 'exact byte SHA256 for same-platform comparison'}}
    (ROOT / 'tests/fixtures/parameter-contract.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'declared': len(declared), 'captured': len(captured), 'missing': 4, 'additional_SOE': 2}))


if __name__ == '__main__':
    main()
