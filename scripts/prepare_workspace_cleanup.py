"""Prepare exact cleanup candidates for review. This script never deletes."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep-build', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError('Output must remain in v3')
    tracked = set(filter(None, subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')))
    candidates, files, covered = [], [], set()
    groups = defaultdict(lambda: {'bytes': 0, 'files': 0, 'targets': 0})

    def add(path, group, reason):
        if not path.exists():
            return
        path = path.resolve(strict=True)
        if not path.is_relative_to(ROOT) or path == ROOT:
            raise ValueError('Target escaped v3')
        relative = path.relative_to(ROOT).as_posix()
        if relative in tracked or any(p.startswith(relative + '/') for p in tracked):
            raise ValueError('Tracked content is excluded: ' + relative)
        paths = [path] if path.is_file() else path.rglob('*')
        rows = []
        for p in paths:
            stat = p.lstat()
            if p.is_symlink() or getattr(stat, 'st_file_attributes', 0) & 0x400:
                raise ValueError('Links are excluded: ' + str(p))
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT).as_posix()
            if rel in covered:
                raise ValueError('Overlapping candidate: ' + rel)
            covered.add(rel)
            rows.append({'path': rel, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                         'hardlinks': stat.st_nlink, 'group': group})
        size = sum(p['bytes'] for p in rows)
        files.extend(rows)
        candidates.append({'path': relative, 'absolute_path': str(path), 'group': group,
                           'kind': 'file' if path.is_file() else 'directory',
                           'bytes': size, 'files': len(rows), 'reason': reason})
        groups[group]['bytes'] += size
        groups[group]['files'] += len(rows)
        groups[group]['targets'] += 1

    # Preserve snapshots, build logs, specs, TOCs and size/hash audit reports.
    for build in (ROOT / 'output').iterdir():
        if not build.is_dir() or not (build.name == 'build' or build.name.startswith('build-')):
            continue
        if build.name == 'build-' + args.keep_build:
            continue
        for p in build.rglob('*'):
            if p.is_file() and 'snapshot' not in p.relative_to(build).parts and (
                p.suffix.lower() in {'.pkg', '.pyz'} or p.name == 'base_library.zip'
            ):
                add(p, 'old_build_archives', 'Old PyInstaller intermediate; rebuildable; source snapshot and logs retained')
    add(ROOT / 'output/p01-probe/PhoneticToolbox-P01.exe', 'old_probe', 'Obsolete P01 executable; hash record retained')
    for rel in ('dist/research-repair/PhoneticToolbox-v3-Research-Fix1.exe',
                'dist/PhoneticToolbox-v3-Latest-20261003/PhoneticToolbox-v3-Latest-20261003.exe'):
        add(ROOT / rel, 'obsolete_executables', 'Old trial or failed QA candidate; final R1 retained')
    component = ROOT / 'output/validation/m11/component-a4159343dbc44af180e2f2e361666ef9'
    add(component / 'installed', 'mfa_test_copies', 'Failed MFA candidate test install; actual registered output/m11c-028b881d retained')
    add(component / 'mfa-3.3.8-windows-x86_64-candidate.zip', 'mfa_test_copies', 'Generated candidate archive, separate from active registered runtime')
    for test in (ROOT / 'output/validation/m11').iterdir():
        if test.is_dir() and test.name.startswith(('wiring-', 'qt-')):
            add(test / 'components', 'mfa_test_copies', 'Isolated test component copies; reports and actual registered runtime retained')
    for test in (ROOT / 'output/validation/m03-runtime').iterdir():
        if test.is_dir():
            add(test / 'payload', 'm03_staged_copy', 'Staging probe runtime payload; active .venv/m03-compatible retained')
    for test in (ROOT / 'output/validation/m16').iterdir():
        if test.is_dir() and test.name.startswith('long-'):
            report = json.loads((test / 'report.json').read_text('utf-8'))
            if report.get('success') is True and 'accelerated synthetic' in report.get('scope', ''):
                add(test / 'project', 'synthetic_long_recordings', 'Verified generated synthetic 60-minute recording; SHA-256 report retained')
    # Hash the small retained reports associated with selected validation payloads.
    evidence = {}
    for target in candidates:
        path = ROOT / target['path']
        if not target['path'].startswith('output/validation/'):
            continue
        for name in ('report.json', 'receipt.json', 'status.json', 'request.json', 'response.json'):
            p = path.parent / name
            if p.is_file() and p.stat().st_size <= 10 * 1024 * 1024:
                evidence[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    report = {'status': 'prepared_only_no_deletion', 'requires': 'Explicit approval of these targets and verified final R1 executable',
              'root': str(ROOT), 'keep_build': args.keep_build,
              'bytes': sum(v['bytes'] for v in groups.values()), 'groups': dict(groups),
              'targets': candidates, 'files': files, 'retained_evidence_sha256': evidence,
              'preserve': ['output/m11c-028b881d', '.venv', 'source snapshots, specs, TOCs, logs, reports',
                           'dist/' + args.keep_build,
                           'real device media, comparison baselines, databases, research data',
                           'V2 checkout and shared Git history'],
              'note': 'Logical byte totals are upper bounds on reclaimed space when hardlinks exist; recheck size/mtime and active processes before any future deletion'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'bytes', 'groups')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
