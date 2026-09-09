"""Capture original-v2 reference processes without altering source, environment or research data."""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

from baseline_support import RECIPES, compare, create_fixture, load_json, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-egg', action='store_true', help='Requires a confirmed local EGG selection file')
    parser.add_argument('--freeze-public', action='store_true', help='Explicitly freeze new public baseline after repeat comparison')
    parser.add_argument('--case', action='append', help='Restrict to named case(s)')
    args = parser.parse_args()
    baseline = load_json(ROOT / 'docs/baseline/local-evidence.json')
    context = load_json(ROOT / 'output/validation/p03/context-before.json')
    source = Path(baseline['baseline']['root'])
    python = Path(context['v2_environment']) / 'python.exe'
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output = ROOT / 'output/validation/p03' / run_id
    output.mkdir(parents=True, exist_ok=False)
    fixtures = output / 'inputs'
    fixtures.mkdir()
    cases = []
    for recipe in RECIPES:
        if recipe['mode'] == 'egg' and not args.include_egg:
            continue
        if args.case and recipe['id'] not in args.case:
            continue
        cases.append({'case_id': recipe['id'], 'input': str(create_fixture(fixtures, recipe)), 'mode': recipe['mode'],
                      'privacy': 'public_synthetic', 'flip_channels': False, 'channel_evidence': 'generator declares left shape/right audio'})
    for sample in baseline['environment']['samples']:
        if sample['id'] not in ['LOCAL-01', 'LOCAL-06', 'LOCAL-09', 'LOCAL-12']:
            continue
        if args.case and sample['id'] not in args.case:
            continue
        case = {'case_id': sample['id'], 'input': sample['path'], 'mode': 'acoustic', 'privacy': 'private_local_only'}
        grids = [p for p in sample['sidecars'] if p.lower().endswith('.textgrid')]
        if grids:
            case['textgrid'] = grids[0]
        cases.append(case)
    if args.include_egg:
        confirmed = load_json(ROOT / 'output/validation/p03/confirmed-egg.json')
        for case in confirmed['cases']:
            if not args.case or case['case_id'] in args.case:
                cases.append(dict(case, mode='egg', privacy='private_local_only'))
    if not cases or (args.case and set(args.case) != {c['case_id'] for c in cases}):
        raise ValueError('Unknown or unavailable case selection; EGG cases require --include-egg')
    results = []
    for case in cases:
        paths = [Path(case['input'])] + ([Path(case['textgrid'])] if case.get('textgrid') else [])
        input_hashes = {str(p): sha(p) for p in paths}
        captured = []
        for repeat in [1, 2]:
            folder = output / case['case_id'] / str(repeat)
            folder.mkdir(parents=True)
            temporary = folder / 'tmp'
            temporary.mkdir()
            request = dict(case, source_root=str(source), output=str(folder / 'result.json.gz'), output_root=str(output))
            request_path = folder / 'request.json'
            request_path.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
            env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUTF8': '1', 'TEMP': str(temporary), 'TMP': str(temporary)}
            # Original environment's native dependencies must retain their original lookup semantics.
            prefix = python.parent
            env['PATH'] = os.pathsep.join([str(prefix), str(prefix / 'Library/bin'), str(prefix / 'Scripts'), env.get('PATH', '')])
            command = [str(python), '-B', '-X', 'utf8', '-X', 'faulthandler', str(ROOT / 'scripts/baseline_worker.py'), str(request_path)]
            with (folder / 'worker.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240)
            if result.returncode:
                raise RuntimeError(f'Reference worker failed: {case["case_id"]}; exit={result.returncode}; see {folder / "worker.log"}')
            captured.append(load_json(request['output']))
            print(f'{case["case_id"]} repeat {repeat}: {captured[-1]["scientific"]["status"]}', flush=True)
        differences = compare(captured[0]['scientific'], captured[1]['scientific'])
        assert not differences, (case['case_id'], differences)
        assert all(sha(p) == input_hashes[str(p)] for p in paths), 'Input changed during capture'
        results.append({'id': case['case_id'], 'privacy': case['privacy'],
            'input_sha256': input_hashes[str(paths[0])], 'sidecar_sha256': [input_hashes[str(p)] for p in paths[1:]],
            'status': captured[0]['scientific']['status'], 'repeat_differences': differences,
            'audio': captured[0]['scientific'].get('audio'),
            'channel_selection': captured[0]['scientific'].get('channel_selection'),
            'capture_path': (output / case['case_id'] / '1/result.json.gz').relative_to(ROOT).as_posix()})
    report = {'run_id': run_id, 'cases': results, 'egg_included': args.include_egg, 'status': 'repeat_verified'}
    write_json(output / 'capture-summary.json', report)
    write_json(ROOT / 'output/validation/p03/latest-capture.json', report)
    if args.freeze_public:
        freeze(results)
    print(json.dumps({'run_id': run_id, 'cases': len(results), 'repeat_verified': True}))


def freeze(results):
    folder = ROOT / 'tests/fixtures/golden'
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'tests/fixtures/manifest.json'
    manifest = load_json(manifest_path) if manifest_path.exists() else {
        'schema_version': 1, 'producer': {'kind': 'original-v2-conda-process', 'environment': 'phonetic_311'},
        'v3_algorithm_parity': 'not_implemented', 'public_cases': [], 'private_cases': [],
        'tolerance': {'default_rtol': 1e-7, 'default_atol': 1e-10,
                      'exact': ['keys', 'shape', 'time_axis', 'missing_masks', 'units', 'backend_identity'],
                      'large_egg_preprocessing_arrays': 'same-platform exact byte digest; no cross-platform tolerance claim'}}
    public = {r['id']: r for r in manifest['public_cases']}
    private = {r['id']: r for r in manifest['private_cases']}
    for result in results:
        if result['privacy'] == 'public_synthetic':
            target = folder / (result['id'] + '.json.gz')
            source = ROOT / result['capture_path']
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise RuntimeError('Refusing to replace existing frozen baseline: ' + str(target))
            target.write_bytes(source.read_bytes())
            public[result['id']] = {k: result[k] for k in ['id', 'input_sha256', 'status']}
            public[result['id']].update(baseline_path=target.relative_to(ROOT).as_posix(), baseline_sha256=sha(target))
        else:
            private[result['id']] = {k: result[k] for k in ['id', 'input_sha256', 'sidecar_sha256', 'status', 'privacy']}
        entry = public[result['id']] if result['privacy'] == 'public_synthetic' else private[result['id']]
        scientific = load_json(ROOT / result['capture_path'])['scientific']
        entry['audio'] = scientific.get('audio')
        entry['channel_selection'] = scientific.get('channel_selection')
        entry['requirements'] = ['P03', 'M01' if scientific['mode'] == 'acoustic' else 'M03']
    manifest['public_cases'], manifest['private_cases'] = list(public.values()), list(private.values())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
