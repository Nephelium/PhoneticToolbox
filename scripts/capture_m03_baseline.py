"""Capture M03 references twice in original phonetic_311, with no source mutation."""
import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from baseline_support import RECIPES, create_fixture, load_json, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def fixtures(folder):
    original = create_fixture(folder, next(r for r in RECIPES if r['id'] == 'SYN-EGG-44100'))
    rate, data = wavfile.read(original)
    recipes = {'PCM16': data, 'FLOAT': data.astype(np.float32) / 32767,
               'SWAPPED': data[:, ::-1], 'SILENCE': np.zeros_like(data),
               'MONO': data[:, 0], 'SHORT': data[:8], 'EMPTY': data[:0],
               'MULTI': np.column_stack((data, data[:, 0]))}
    cases = []
    for label, samples in recipes.items():
        case_id = 'EGG-SYN-' + label
        path = folder / (case_id + '.wav')
        wavfile.write(path, rate, samples)
        cases.append({'case_id': case_id, 'input': str(path), 'flip_channels': label == 'SWAPPED',
                      'matrix': label == 'PCM16', 'gui': label == 'PCM16', 'privacy': 'public_synthetic'})
    broken = folder / 'EGG-SYN-BROKEN.wav'
    broken.write_bytes(b'M03 deliberately invalid WAV')
    cases.append({'case_id': 'EGG-SYN-BROKEN', 'input': str(broken), 'privacy': 'public_synthetic'})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-public', action='store_true')
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    baseline = load_json(ROOT / 'docs/baseline/local-evidence.json')
    source = Path(baseline['baseline']['root']).resolve()
    context = load_json(ROOT / 'output/validation/p03/context-before.json')
    python = Path(context['v2_environment']) / 'python.exe'
    assert source == (ROOT.parent / 'PhoneticToolbox_v2').resolve()
    assert python.is_file()
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = ROOT / 'output/validation/m03-baseline' / run_id
    output.mkdir(parents=True)
    inputs = output / 'inputs'
    inputs.mkdir()
    cases = fixtures(inputs)
    confirmed = load_json(ROOT / 'output/validation/p03/confirmed-egg.json')
    p03 = load_json(ROOT / 'tests/fixtures/manifest.json')
    for entry in confirmed['cases']:
        expected = next(c for c in p03['private_cases'] if c['id'] == entry['case_id'])
        assert sha(entry['input']) == expected['input_sha256'], 'Private source differs from approved P03 baseline'
        cases.append(dict(entry, privacy='private_local_only'))
    if args.case:
        cases = [c for c in cases if c['case_id'] in args.case]
        assert {c['case_id'] for c in cases} == set(args.case)
    before = {p.relative_to(source).as_posix(): sha(p) for p in (source / 'phonetic_toolbox').rglob('*.py')}
    records = []
    for case in cases:
        input_hash = sha(case['input'])
        captures = []
        for repeat in (1, 2):
            folder = output / case['case_id'] / str(repeat)
            folder.mkdir(parents=True)
            temporary = folder / 'tmp'
            temporary.mkdir()
            request = dict(case, source_root=str(source), output=str(folder), output_root=str(output))
            write_json(folder / 'request.json', request)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
                       TEMP=str(temporary), TMP=str(temporary), QT_QPA_PLATFORM='offscreen',
                       MPLCONFIGDIR=str(folder / 'mplconfig'))
            prefix = python.parent
            env['PATH'] = os.pathsep.join([str(prefix), str(prefix / 'Library/bin'), str(prefix / 'Scripts'), env.get('PATH', '')])
            command = [str(python), '-B', '-X', 'utf8', '-X', 'faulthandler',
                       str(ROOT / 'scripts/m03_baseline_worker.py'), str(folder / 'request.json')]
            with (folder / 'worker.log').open('w', encoding='utf-8') as log:
                completed = subprocess.run(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=240, creationflags=subprocess.CREATE_NO_WINDOW)
            if completed.returncode:
                raise RuntimeError(f'{case["case_id"]} worker failed ({completed.returncode}); {folder / "worker.log"}')
            captures.append(load_json(folder / 'result.json'))
            print(f'{case["case_id"]} repeat {repeat}: {captures[-1]["load"]["status"]}', flush=True)
        assert captures[0] == captures[1], f'Exact metadata/array/pixel repeat mismatch: {case["case_id"]}'
        assert sha(case['input']) == input_hash, 'Input modified'
        first = output / case['case_id'] / '1'
        records.append({'id': case['case_id'], 'privacy': case['privacy'], 'input_sha256': input_hash,
                        'repeat_equal': True, 'input_unchanged': True,
                        'metadata': case['case_id'] + '.json', 'arrays': case['case_id'] + '.npz',
                        'metadata_sha256': sha(first / 'result.json'), 'arrays_sha256': sha(first / 'arrays.npz'),
                        'capture': first.relative_to(ROOT).as_posix()})
    after = {p.relative_to(source).as_posix(): sha(p) for p in (source / 'phonetic_toolbox').rglob('*.py')}
    assert before == after, 'Original v2 source changed'
    manifest = {'schema_version': 1, 'producer': 'original-v2-conda-process',
                'repeat_comparison': 'exact_arrays_masks_and_metadata', 'run_id': run_id,
                'source_unchanged': True, 'source_file_count': len(before),
                'public_cases': [r for r in records if r['privacy'] == 'public_synthetic'],
                'private_cases': [r for r in records if r['privacy'] == 'private_local_only']}
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'source-hashes.json', before)
    write_json(ROOT / 'output/validation/m03-baseline/latest.json', manifest)
    if args.freeze_public:
        assert not args.case, 'Freeze only complete capture'
        target = ROOT / 'tests/fixtures/m03'
        target.mkdir(exist_ok=True)
        pending = []
        for record in manifest['public_cases']:
            for field, name in [('metadata', 'result.json'), ('arrays', 'arrays.npz')]:
                dest = target / record[field]
                content = (ROOT / record['capture'] / name).read_bytes()
                if dest.exists() and dest.read_bytes() != content:
                    raise RuntimeError('Refusing to replace existing frozen evidence: ' + dest.name)
                pending.append((dest, content))
        manifest_path = target / 'manifest.json'
        if manifest_path.exists():
            raise RuntimeError('Manifest already frozen; use a separately reviewed new baseline version')
        for dest, content in pending:
            dest.write_bytes(content)
        write_json(manifest_path, manifest)
    print(json.dumps({'run_id': run_id, 'cases': len(records), 'source_unchanged': True}), flush=True)


if __name__ == '__main__':
    main()
