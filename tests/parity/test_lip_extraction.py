"""M05-A original V2 frozen oracle; test execution cannot generate expected."""
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/phonetic_core/src'))
from phonetic_core.lip.metrics import extract_lip_metrics
from phonetic_core.lip.stabilizer import LandmarkStabilizer
from phonetic_core.lip.sequence import LipConfig, LipSequence, offset_time

with gzip.open(ROOT / 'tests/fixtures/m05/v2.json.gz', 'rt', encoding='utf-8') as stream:
    ORACLE = json.load(stream)


def test_frozen_source_and_indices():
    import hashlib
    from phonetic_core.lip import metrics
    original_hashes = {key.replace('\\', '/'): value for key, value in ORACLE['producer']['source_hashes'].items()}
    assert hashlib.sha256((ROOT / 'packages/phonetic_core/src/phonetic_core/lip/metrics.py').read_bytes()).hexdigest() == original_hashes['phonetic_toolbox/core/lip/metrics.py']
    spec = json.loads((ROOT / 'resources/m05/metric-spec.json').read_text('utf-8'))
    for name in ('OUTER_LIP_LANDMARKS', 'INNER_LIP_LANDMARKS', 'FACE_OVAL', 'LEFT_FACE', 'RIGHT_FACE', 'TOP_FACE', 'BOTTOM_FACE'):
        assert spec[name] == getattr(metrics, name)
    assert spec['neighbors'] == ORACLE['spec']['neighbors']


@pytest.mark.parametrize('case', ORACLE['scientific'], ids=lambda c: f"{c['case']}-filter-{c['filter']}")
def test_original_video_landmarks_filter_and_metrics_exact(case):
    neighbors = [np.asarray(n, dtype=np.int64) for n in ORACLE['spec']['neighbors']]
    seq = LipSequence(LipConfig(filter_enabled=case['filter']), neighbors)
    for index, raw in enumerate(case['raw']):
        row = seq.process(raw, index / 25)
        assert row['detected'] == (raw is not None)
        if raw is None:
            assert row['metrics'] is None  # Original imputed compatibility values are not measurements.
        else:
            np.testing.assert_array_equal(np.asarray(row['points'], dtype=np.float32), np.asarray(case['landmarks_full'][index], dtype=np.float32))
            for key, value in row['metrics'].items(): assert value == case['metrics'][key][index]


@pytest.mark.parametrize('case', ORACLE['analytic'], ids=lambda c: f"cutoff-{c['cutoff']}")
def test_original_filter_dynamics_exact(case):
    neighbors = [np.asarray(n, dtype=np.int64) for n in ORACLE['spec']['neighbors']]
    stabilizer = LandmarkStabilizer(min_cutoff_hz=case['cutoff'], neighbor_indices=neighbors)
    for i, raw in enumerate(ORACLE['input_points']):
        result = stabilizer.filter(np.asarray(raw, dtype=np.float32), i * .043)
        np.testing.assert_array_equal(result, np.asarray(case['filtered'][i], dtype=np.float32))
        actual = extract_lip_metrics(result)
        for key, expected in case['metrics'][i].items():
            if expected is None: assert np.isnan(actual[key])
            else: assert actual[key] == expected


def test_vfr_never_uses_frame_index():
    seq = LipSequence(LipConfig(False))
    times = [.023, .067, .2, .221]
    rows = [seq.process(None, t) for t in times]
    assert [r['time_s'] for r in rows] == times
    assert all(not r['detected'] and r['points'] is None for r in rows)
    with pytest.raises(ValueError, match='nonmonotonic'): seq.process(None, times[-1])


def test_offset_actions_and_negative_open():
    assert offset_time(.05, -.1) == -.05
    assert offset_time(.05, -.1, 'save_without_offset') == .05
    assert offset_time(.05, -.1, 'cancel') is None
    for value in (float('nan'), float('inf'), 2.01):
        with pytest.raises(ValueError): offset_time(0, value)
    points = np.asarray(ORACLE['input_points'][4], dtype=np.float32)
    assert extract_lip_metrics(points)['open_px'] == -1


def test_all_degenerate_and_input_not_mutated():
    p = np.zeros((478, 2), dtype=np.float32)
    before = p.copy()
    result = extract_lip_metrics(p)
    assert np.isnan(result['area']) and np.isnan(result['circularity'])
    assert result['open_px'] == 0
    np.testing.assert_array_equal(p, before)
