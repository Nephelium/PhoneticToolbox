"""Compare a native oracle and Wasm using the same synthetic recipes."""
import argparse
import base64
import json
from pathlib import Path
import numpy as np
from scipy.signal import welch


def compare(native, wasm):
    report = {'rtol': 1e-7, 'atol': 1e-10, 'geometry': {}, 'audio': {},
              'timings_ms': {'native': native['timings'], 'wasm': wasm['timings']}}
    failures = []
    stats = {'numbers': 0, 'max_abs': 0., 'failed_arrays': 0}

    def check(a, b, path):
        if isinstance(a, dict):
            if a.keys() != b.keys():
                failures.append({'path': path, 'issue': 'keys'})
                return
            for key in a:
                check(a[key], b[key], path + '/' + key)
        elif isinstance(a, list):
            if len(a) != len(b):
                failures.append({'path': path, 'issue': 'length'})
            elif a and all(isinstance(v, (float, int)) for v in a):
                x, y = np.array(a), np.array(b)
                error = float(np.max(np.abs(x-y)))
                stats['numbers'] += len(x)
                stats['max_abs'] = max(stats['max_abs'], error)
                if not np.allclose(x, y, rtol=1e-7, atol=1e-10):
                    stats['failed_arrays'] += 1
                    if len(failures) < 40:
                        failures.append({'path': path, 'max_abs': error})
            else:
                for index, (x, y) in enumerate(zip(a, b)):
                    check(x, y, path + '/' + str(index))
        elif isinstance(a, (float, int)) and not isinstance(a, bool):
            check([a], [b], path)
        elif a != b:
            failures.append({'path': path, 'issue': 'value'})

    for field in ('sample_rate', 'geometry_method', 'parameters', 'isolated_libraries', 'snapshots', 'tubes', 'transfer_raw', '33/sequence', '65/sequence'):
        check(native[field], wasm[field], field)
    report['strict_numeric_audit'] = {**stats, 'pass': not failures, 'failures': failures}
    # Keep the initial strict audit, and separately describe geometry and the
    # explicitly accepted tiny cross-compiler transfer-function differences.
    failures = []
    stats = {'numbers': 0, 'max_abs': 0., 'failed_arrays': 0}
    for field in ('sample_rate', 'geometry_method', 'parameters', 'isolated_libraries', 'tubes', '33/sequence', '65/sequence'):
        check(native[field], wasm[field], field)
    for key, item in native['snapshots'].items():
        check({k:v for k,v in item.items() if k!='transfer_db'},
              {k:v for k,v in wasm['snapshots'][key].items() if k!='transfer_db'}, 'snapshots/' + key)
    report['geometry'] = {**stats, 'pass': not failures, 'failures': failures}
    display_difference = max(float(np.max(np.abs(np.array(item['transfer_db'])-wasm['snapshots'][key]['transfer_db'])))
                             for key, item in native['snapshots'].items())
    raw_db_difference = max(float(np.max(np.abs(20*np.log10(np.maximum(item['magnitude'],1e-10))-
                      20*np.log10(np.maximum(wasm['transfer_raw'][key]['magnitude'],1e-10)))))
                      for key,item in native['transfer_raw'].items())
    report['transfer'] = {'max_display_difference_db': display_difference, 'max_raw_difference_db': raw_db_difference,
        'max_phase_difference_rad': max(float(np.max(np.abs(np.array(item['phase'])-wasm['transfer_raw'][key]['phase'])))
            for key,item in native['transfer_raw'].items()),
        'resonance_grids_identical': all(item['resonances']==wasm['snapshots'][key]['resonances'] for key,item in native['snapshots'].items()),
        'user_accepted_display_difference_db': .001,
        'accepted_display_difference': display_difference<=.001+1e-12 and raw_db_difference<=.001}
    for key, item in native['audio'].items():
        x = np.frombuffer(base64.b64decode(item['base64']), dtype='<f8')
        y = np.frombuffer(base64.b64decode(wasm['audio'][key]['base64']), dtype='<f8')
        assert x.shape == y.shape
        f, p = welch(x, fs=48000, nperseg=min(2048, len(x)))
        _, q = welch(y, fs=48000, nperseg=min(2048, len(y)))
        bands = {}
        for low, high in ((0, 500), (500, 1500), (1500, 3000), (3000, 6000), (6000, 12000)):
            selected = (f >= low) & (f < high)
            bands[f'{low}-{high}'] = float(10*np.log10(max(float(q[selected].sum()), 1e-30)/max(float(p[selected].sum()), 1e-30)))
        report['audio'][key] = {'samples': len(x), 'finite': bool(np.isfinite(y).all()),
            'pcm16_max_lsb': int(np.max(np.abs(np.rint(np.clip(x,-1,1)*32767)-np.rint(np.clip(y,-1,1)*32767)))),
            'max_abs_difference': float(np.max(np.abs(x-y))), 'native_rms': item['rms'], 'wasm_rms': wasm['audio'][key]['rms'],
            'rms_difference_db': float(20*np.log10(max(wasm['audio'][key]['rms'], 1e-30)/max(item['rms'], 1e-30))),
            'correlation': float(np.corrcoef(x, y)[0, 1]), 'band_power_difference_db': bands,
            'native_peak_hz': float(f[np.argmax(p)]), 'wasm_peak_hz': float(f[np.argmax(q)])}
    report['acceptance'] = {'geometry_and_timing': report['geometry']['pass'],
        'tiny_transfer_difference_accepted_by_user': report['transfer']['accepted_display_difference'] and report['transfer']['resonance_grids_identical'],
        'zero_jitter_diagnostic_within_one_pcm16_lsb': all(value['pcm16_max_lsb']<=1 for key,value in report['audio'].items() if key.startswith('diagnostic')),
        'all_audio_finite_and_sample_counts_identical': all(value['finite'] for value in report['audio'].values()),
        'production_jitter': 'Original 10 percent JD2 jitter and native rand() retained. Platform random trajectories differ; no promise of sample-identical voiced production audio.'}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--wasm', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.native.read_text('utf-8')), json.loads(args.wasm.read_text('utf-8')))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
