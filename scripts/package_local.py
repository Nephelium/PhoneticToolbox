"""One local-only workflow; delegates EXE freezing to the authoritative builder."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def frontend_inputs():
    files = []
    for relative in ('frontend/src', 'frontend/public'):
        files.extend(p for p in (ROOT / relative).rglob('*') if p.is_file()
                     and not p.is_relative_to(ROOT / 'frontend/public/manual'))
    files.extend(ROOT / p for p in ('frontend/index.html', 'frontend/feedback-manager.html',
                 'frontend/package.json', 'frontend/package-lock.json', 'frontend/vite.config.ts', 'manual/project.json'))
    files.extend((ROOT / 'manual/chapters').glob('*.json'))
    result = {}
    for path in sorted(files):
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('Frontend input escapes checkout')
        with path.open('rb') as stream:
            result[path.relative_to(ROOT).as_posix()] = {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    return result


def select_cache(root: Path) -> Path:
    candidates = []
    for work in (root / 'output').glob('build-*'):
        try:
            lifecycle = json.loads((work / 'artifact-lifecycle.json').read_text('utf8'))
            info = json.loads((root / 'dist' / work.name.removeprefix('build-') / 'build-info.json').read_text('utf8'))
            bundle = json.loads((work / 'snapshot/desktop-bundle.json').read_text('utf8'))
            if (lifecycle.get('state') == 'completed' and info.get('portable') and info.get('archives')
                    and not bundle.get('mfa') and set(bundle['runtimes']) == {'PTB_EGG_PYTHON', 'PTB_M05_PYTHON'}
                    and (work / 'host-archive/host-archive.json').is_file()):
                candidates.append((lifecycle['finished_at'], work))
        except (OSError, ValueError, KeyError):
            continue
    if not candidates:
        raise ValueError('No completed finalized compact cache; prepare a runtime stage with the documented lower-level builder first')
    return max(candidates)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True, help='Requested local app/core/frontend version, e.g. 3.0.2')
    parser.add_argument('--plan', action='store_true', help='Show paths and cache candidate without writing anything')
    args = parser.parse_args()
    if not re.fullmatch(r'\d+\.\d+\.\d+(?:-[A-Za-z0-9.]+)?', args.version):
        parser.error('Use a semantic release version')
    cache = select_cache(ROOT)
    name = 'PhoneticToolbox-' + args.version.replace('.', '') + '-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    run = ROOT / 'output/packaging' / name
    package = ROOT / 'dist' / name
    work = ROOT / 'output' / ('build-' + name)
    plan = {'version': args.version, 'name': name, 'runtimeCache': str(cache / 'snapshot'),
            'hostCache': str(cache / 'host-archive'), 'mediaCache': str(ROOT / 'output/cache/manual-media'),
            'evidence': str(run), 'package': str(package), 'upload': False}
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if sys.platform != 'win32':
        parser.error('Windows build environment required')
    expected_python = ROOT / '.venv/m14/Scripts/python.exe'
    if Path(sys.executable).absolute() != expected_python.absolute():
        parser.error('Run with .venv/m14/Scripts/python.exe')
    # Prevent concurrent workflows from changing version metadata mid-freeze.
    lock = ROOT / 'output/package-local.lock'
    lock.parent.mkdir(exist_ok=True)
    with lock.open('x', encoding='utf8') as stream:
        stream.write(json.dumps({'ownerPid': os.getpid(), 'run': str(run)}))
    start = time.perf_counter()
    report = plan | {'schema': 'ptb-local-package/1', 'state': 'active', 'startedAt': datetime.now(timezone.utc).isoformat(), 'phases': []}
    run.mkdir(parents=True, exist_ok=False)

    def save():
        report['elapsedSeconds'] = round(time.perf_counter() - start, 3)
        (run / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf8')

    def phase(label, command, cwd=ROOT, env=None):
        print('Starting ' + label, flush=True)
        began = time.perf_counter()
        log = run / (label + '.log')
        with log.open('wb') as stream:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        report['phases'].append({'phase': label, 'command': list(map(str, command)), 'log': str(log),
                                 'seconds': round(time.perf_counter() - began, 3), 'returncode': result.returncode})
        save()
        if result.returncode:
            print(log.read_text('utf8', errors='replace')[-5000:], flush=True)
            raise subprocess.CalledProcessError(result.returncode, command)
        print(label + ': ' + str(report['phases'][-1]['seconds']) + 's', flush=True)

    python = [sys.executable, '-B', '-X', 'utf8']
    npm = shutil.which('npm.cmd')
    node = shutil.which('node')
    try:
        if not npm or not node:
            raise ValueError('Existing Node/npm are required; this workflow does not install dependencies')
        report['sourceCommit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        (run / 'working-tree.txt').write_text(subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True), 'utf8')
        version_path = ROOT / 'release/version.json'
        version = json.loads(version_path.read_text('utf8'))
        version.update({key: args.version for key in ('app_version', 'core_version', 'frontend_version')})
        version_path.write_text(json.dumps(version, indent=2) + '\n', 'utf8')
        phase('versions', python + ['scripts/sync_versions.py'])
        inputs = frontend_inputs()
        (run / 'frontend-inputs.json').write_text(json.dumps(inputs, indent=2) + '\n', 'utf8')
        phase('manual-validate', python + ['scripts/manual/validate.py', '--project', 'manual', '--strict'])
        public = run / 'public'
        shutil.copytree(ROOT / 'frontend/public', public, ignore=lambda folder, names: ['manual'] if Path(folder) == ROOT / 'frontend/public' else [])
        phase('manual-build', python + ['scripts/manual/build.py', '--project', 'manual', '--output', str(public / 'manual'), '--distribution', 'software'])
        for label, script in [('typecheck', 'typecheck'), ('frontend-tests', 'test'), ('ui-data', 'ui-data:check')]:
            phase(label, [npm, '--prefix', 'frontend', 'run', script])
        # Vite loads the normal configuration, replacing only generated data paths.
        env = os.environ | {'PTB_BUILD_PUBLIC': str(public), 'PTB_BUILD_DIST': str(run / 'frontend-dist')}
        phase('frontend-build', [node, '--input-type=module', '--eval',
              'import {build} from "vite"; await build({publicDir:process.env.PTB_BUILD_PUBLIC,build:{outDir:process.env.PTB_BUILD_DIST}});'],
              cwd=ROOT / 'frontend', env=env)
        if frontend_inputs() != inputs:
            raise RuntimeError('Frontend/manual source changed while generating; start a fresh build')
        phase('exe-build', python + ['scripts/build_v3_local_preview.py', '--name', name, '--lean-qt', '--compact-onefile',
              '--persistent-cache', '--frontend-dist', str(run / 'frontend-dist'), '--runtime-bundle-cache', str(cache / 'snapshot'),
              '--host-archive-cache', str(cache / 'host-archive'), '--manual-media-cache', plan['mediaCache']])
        phase('finalize', python + ['release/finalize_compact.py', '--package', str(package), '--work', str(work)])
        phase('package-audit', python + ['scripts/audit_release_contents.py', str(package / 'PhoneticToolbox-windows-x64.exe'),
              '--output', str(run / 'package-audit.json')])
        # The lower builder keeps a cache while its own command line references
        # it. Retire old completed outputs only after that child has exited.
        first_cleanup = package / 'cleanup-report.json'
        if first_cleanup.exists():
            shutil.copyfile(first_cleanup, run / 'builder-cleanup-report.json')
        phase('cleanup', ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
              str(ROOT / 'release/cleanup_old_builds.ps1'), '-KeepBuild', str(package)])
        report['artifact'] = json.loads((package / 'build-info.json').read_text('utf8'))
        media = json.loads((work / 'manual-package-selection.json').read_text('utf8'))
        report['media'] = {'included': media['included'], 'cacheHits': sum(row.get('cacheHit', False) for row in media['media'])}
        report['state'] = 'built'
        report['verification'] = 'Static identity/audit and source checks only; final EXE execution acceptance remains separate'
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        raise
    finally:
        try:
            save()
        finally:
            lock.unlink()
    print(json.dumps({'exe': str(package / report['artifact']['exe']), 'report': str(run / 'report.json'),
                      'seconds': report['elapsedSeconds'], 'media': report['media']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
