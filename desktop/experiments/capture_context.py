"""P01 read-only preservation evidence; never copies user data or environments."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(root), *args]).decode('utf-8').strip()


def capture() -> dict:
    evidence = json.loads((ROOT / 'docs/baseline/local-evidence.json').read_text('utf-8'))
    original = Path(evidence['baseline']['root'])
    files = evidence['baseline']['files']
    changed = [item['path'] for item in files if not (original / item['path']).is_file()
               or sha(original / item['path']) != item['sha256']]
    envs = json.loads(subprocess.check_output(['conda.bat', 'env', 'list', '--json']))['envs']
    old_env = next(Path(p) for p in envs if Path(p).name == 'phonetic_311')
    packages = subprocess.check_output([str(old_env / 'python.exe'), '-X', 'utf8', '-c',
        'import importlib.metadata,json; print(json.dumps(sorted((d.metadata["Name"],d.version) for d in importlib.metadata.distributions())))'])
    index = Path(git(original, 'rev-parse', '--git-path', 'index'))
    if not index.is_absolute():
        index = original / index
    return {'v2_head': git(original, 'rev-parse', 'HEAD'),
            'v2_index_sha256': sha(index),
            'v2_status': git(original, 'status', '--porcelain=v1', '-z'),
            'baseline_files_checked': len(files), 'baseline_files_changed_since_D03': changed,
            'v2_package_metadata_sha256': hashlib.sha256(packages).hexdigest(),
            'v2_environment': str(old_env),
            'v3_head': git(ROOT, 'rev-parse', 'HEAD'),
            'v3_status': git(ROOT, 'status', '--porcelain=v1')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['before', 'after'])
    args = parser.parse_args()
    destination = ROOT / 'output/validation/p01'
    destination.mkdir(parents=True, exist_ok=True)
    result = capture()
    (destination / f'context-{args.phase}.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.phase == 'after':
        before = json.loads((destination / 'context-before.json').read_text('utf-8'))
        checks = {key: result[key] == before[key] for key in result if key.startswith('v2_')}
        result['preservation_checks'] = checks
        (destination / 'preservation.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
        assert all(checks.values()), checks
    print(json.dumps({key: value for key, value in result.items()
                      if key in ['baseline_files_checked', 'baseline_files_changed_since_D03', 'preservation_checks']}, ensure_ascii=False))
