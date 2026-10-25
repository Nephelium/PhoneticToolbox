"""Run existing final-byte gates locally, including visible GPU and long-path uninstall.

No product registration, uploads, source pushes or real user state are used.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
from owned_qa_paths import DEFAULT_QA_ROOT, qa_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--qa-root', type=Path, default=DEFAULT_QA_ROOT)
    parser.add_argument('--resume', type=Path, help='Resume a failed owned report, preserving passed gates for identical final bytes')
    args = parser.parse_args()
    package = args.package.resolve()
    if package.parent != ROOT / 'dist':
        parser.error('Expected a completed owned dist directory')
    info = json.loads((package / 'build-info.json').read_text('utf8'))
    exe = package / info['exe']
    with exe.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != info['sha256']:
            raise ValueError('Final EXE identity changed')
    token = args.resume.resolve().parent.name.removeprefix('local-package-') if args.resume else uuid.uuid4().hex[:8]
    if not len(token) == 8 or any(c not in '0123456789abcdef' for c in token):
        parser.error('Unexpected owned resume identity')
    out = ROOT / 'output/validation' / ('local-package-' + token)
    selected_qa_root = qa_root(args.qa_root)
    external = selected_qa_root / ('local-' + token)
    out.mkdir(parents=True, exist_ok=bool(args.resume))
    external.mkdir(parents=True, exist_ok=bool(args.resume))
    report = {'schema': 'ptb-local-acceptance/1', 'success': False, 'exeSha256': info['sha256'],
              'version': info['version'], 'package': str(package), 'external': str(external), 'qaRoot': str(selected_qa_root), 'phases': [],
              'scope': 'Final Windows x64 bytes on this computer, private state, default GPU visible window; no physical devices or separate clean computer'}
    if args.resume:
        if args.resume.resolve() != out / 'report.json':
            parser.error('Expected an owned local acceptance report')
        previous = json.loads(args.resume.read_text('utf8'))
        if previous.get('external') != str(external):
            raise ValueError('Resume must use the original QA root')
        if previous.get('exeSha256') != info['sha256'] or previous.get('package') != str(package) or previous.get('success'):
            raise ValueError('Cannot reuse a report for different final bytes or an already successful run')
        shutil.copyfile(args.resume, out / ('failed-report-' + uuid.uuid4().hex[:8] + '.json'))
        report['phases'] = previous['phases']
        # These gates require the same live window. Preserve earlier captures,
        # then rerun the complete affected visible scope in a fresh owned profile.
        preserved = out / ('before-resume-' + uuid.uuid4().hex[:8])
        preserved.mkdir()
        for label in ('visible', 'redundant-work', 'reader-responsiveness'):
            if (out / label).exists():
                shutil.copytree(out / label, preserved / label)
        report['resumeEvidence'] = previous.get('resumeEvidence', []) + [str(preserved)]
    # Older reports stored handoff copies beside the report. Keep those exact
    # paths when resuming; new runs keep all large mutable copies on qa-root.
    handoff_root = out if args.resume and any((out / (kind + '-origin')).exists() for kind in ('portable', 'installer')) else external
    python = [sys.executable, '-B', '-X', 'utf8']
    node = shutil.which('node')
    def save():
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf8')
    def phase(label, command, env=None):
        if args.resume and label not in {'visible-rendering','redundant-work','reader-responsiveness','rendered-pixels'} and any(p['phase'] == label and p['returncode'] == 0 for p in report['phases']):
            print('Preserving passed gate for identical EXE: ' + label, flush=True)
            return
        print('Checking ' + label, flush=True)
        began = time.perf_counter()
        if (out / (label + '.log')).exists():
            shutil.copyfile(out / (label + '.log'), out / (label + '-previous-' + uuid.uuid4().hex[:6] + '.log'))
        with (out / (label + '.log')).open('wb') as stream:
            code = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        report['phases'].append({'phase': label, 'command': list(map(str, command)), 'returncode': code,
                                'seconds': round(time.perf_counter() - began, 3)})
        save()
        if code:
            print((out / (label + '.log')).read_text('utf8', errors='replace')[-6000:], flush=True)
            raise subprocess.CalledProcessError(code, command)
        print(label + ' passed', flush=True)
    def port():
        with socket.socket() as server:
            server.bind(('127.0.0.1', 0))
            return server.getsockname()[1]
    save()
    try:
        phase('distribution', python + ['release/verify_compact_launch.py', '--package', package, '--output', external / 'distribution', '--qa-root', selected_qa_root])
        compiler = ROOT / 'output/release-tools/inno-7.1.0/compiler/ISCC.exe'
        setup_dir = out / 'owned-setup'
        setup_dir.mkdir(exist_ok=bool(args.resume))
        phase('compile-owned-installer', [compiler, '/DCompactOnefile', '/DPersistentCache', '/DOwnedQA',
              '/DAppIdentity={{' + str(uuid.uuid4()).upper() + '}', '/DPackageDir=' + str(package),
              '/DReleaseDir=' + str(setup_dir), '/DAppExecutable=' + exe.name, '/DVersion=' + info['version'], ROOT / 'release/installer.iss'])
        setup = setup_dir / ('PhoneticToolbox-' + info['version'] + '-windows-x64-setup.exe')
        with setup.open('rb') as stream:
            installer_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        report['installer'] = {'path': str(setup), 'sha256': installer_sha,
                               'scriptSha256': hashlib.sha256((ROOT / 'release/installer.iss').read_bytes()).hexdigest()}
        save()
        phase('long-path-install-uninstall', python + ['scripts/verify_cached_install.py', '--setup', setup, '--exe', exe,
              '--output', external / 'long-install', '--with-workspace', '--long-paths', '--qa-root', selected_qa_root])
        phase('cache-processes', python + ['scripts/verify_cache_processes.py', '--exe', exe, '--output', external / 'cache-processes', '--qa-root', selected_qa_root])
        cache_settings = external / 'cache-settings'
        if args.resume and cache_settings.exists():
            cache_settings = external / ('cache-settings-retry-' + uuid.uuid4().hex[:8])
        phase('cache-settings', [node, ROOT / 'scripts/verify_cache_settings.mjs', exe, cache_settings])
        # Real portable and installed handoff use the same final application.
        archive = out / 'owned-update.zip'
        if not archive.exists():
            with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as stream:
                stream.write(exe, 'PhoneticToolbox/PhoneticToolbox.exe')
                stream.write(package / 'application.json', 'PhoneticToolbox/application.json')
        else:
            with zipfile.ZipFile(archive) as stream, stream.open('PhoneticToolbox/PhoneticToolbox.exe') as app_bytes:
                if hashlib.file_digest(app_bytes, 'sha256').hexdigest() != info['sha256']:
                    raise ValueError('Resumed update ZIP differs')
        for kind, delivery in [('portable', archive), ('installer', setup)]:
            origin_folder = handoff_root / (kind + '-origin')
            origin_folder.mkdir(exist_ok=bool(args.resume))
            origin = origin_folder / 'PhoneticToolbox.exe'
            shutil.copyfile(exe, origin)
            shutil.copyfile(package / 'application.json', origin_folder / 'application.json')
            if kind == 'installer':
                seed = external / 'installer-origin-prepare'
                seed.mkdir(exist_ok=bool(args.resume))
                seed_env = {k: v for k, v in os.environ.items() if not k.startswith(('PTB_', 'PYTHON', 'CONDA', '_PYI', 'QT_')) and k != 'VIRTUAL_ENV'}
                seed_env.update(LOCALAPPDATA=str(seed), TEMP=str(seed), TMP=str(seed),
                                PATH=os.pathsep.join((os.environ['SystemRoot'] + '/System32', os.environ['SystemRoot'])),
                                PTB_OWNED_BOOTSTRAP_LOG=str(seed / 'bootstrap-error.log'))
                phase('installer-origin-install', [setup, '/CURRENTUSER', '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-',
                      '/DIR=' + str(origin_folder), '/LOG=' + str(out / 'installer-origin.inno.log')], env=seed_env)
                if not (origin_folder / '.ptb-installed.json').is_file():
                    raise ValueError('Actual installed origin has no installation identity')
            evidence = handoff_root / (kind + '-handoff')
            passed = next((p for p in reversed(report['phases']) if p['phase'] == kind + '-handoff' and p['returncode'] == 0), None)
            if args.resume and passed:
                evidence = Path(passed['command'][passed['command'].index('--output') + 1])
            elif args.resume and any(p['phase'] == kind + '-handoff' and p['returncode'] != 0 for p in report['phases']):
                evidence = handoff_root / (kind + '-handoff-retry-' + uuid.uuid4().hex[:6])
            phase(kind + '-handoff', python + ['scripts/verify_update_handoff.py', '--kind', kind, '--package', delivery,
                  '--origin', origin, '--output', evidence, '--port', str(port()), '--target-version', info['version'],
                  '--current-version', '3.0.1', '--qa-root', selected_qa_root, '--frontend-dist',
                  ROOT / 'output' / ('build-' + package.name) / 'snapshot/frontend/dist'])
            phase(kind + '-relaunch', [node, ROOT / 'scripts/verify_update_relaunch.mjs', evidence / 'handoff.json'])
        # Launch the EXE alone outside the source tree, with normal Windows GPU.
        visible = external / ('visible-retry-' + uuid.uuid4().hex[:8] if args.resume else 'visible')
        visible.mkdir()
        app = visible / exe.name
        shutil.copyfile(exe, app)
        companion = package / 'geolinguistics-data'
        if companion.is_dir():
            shutil.copytree(companion, visible / 'geolinguistics-data')
            report['dataManifestSha256'] = hashlib.sha256((companion / 'data-manifest.json').read_bytes()).hexdigest()
        profile, temp = visible / 'profile', visible / 'temp'
        profile.mkdir(); temp.mkdir()
        from portable_cache_qa import seed_portable_profile
        seed_portable_profile(profile)
        updates = profile / 'PhoneticToolbox/v3/updates'
        updates.mkdir(parents=True,exist_ok=True)
        (updates / 'state.json').write_text('{"autoCheck":false}', 'utf8')
        # This owned PDF exercises cache reuse in the actual final application.
        from pypdf import PdfWriter
        library = visible / 'library'
        library.mkdir()
        pdf = PdfWriter()
        for _ in range(4):
            pdf.add_blank_page(width=612, height=792)
        with (library / 'Owned performance fixture.pdf').open('wb') as stream:
            pdf.write(stream)
        papers = profile / 'PhoneticToolbox/v3/papers'
        papers.mkdir(parents=True)
        (papers / 'preferences.json').write_text(json.dumps(dict(autoReceive=False, libraryRoot=str(library), checkIntervalHours=6)), 'utf8')
        env = {k: v for k, v in os.environ.items() if not k.startswith(('PTB_', 'PYTHON', 'CONDA', '_PYI', 'QT_')) and k != 'VIRTUAL_ENV'}
        debug_port = port()
        env.update(LOCALAPPDATA=str(profile), TEMP=str(temp), TMP=str(temp),
                   PATH=os.pathsep.join((os.environ['SystemRoot'] + '/System32', os.environ['SystemRoot'])),
                   QT_QPA_PLATFORM='windows', QTWEBENGINE_REMOTE_DEBUGGING=str(debug_port),
                   QTWEBENGINE_CHROMIUM_FLAGS='--mute-audio --remote-allow-origins=*',
                   PTB_OWNED_BOOTSTRAP_LOG=str(visible / 'bootstrap-error.log'))
        with (visible / 'process.log').open('wb') as log:
            process = subprocess.Popen([str(app)], env=env, cwd=visible, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                report['visibleLaunch'] = {'pid': process.pid, 'command': [str(app)], 'flags': env['QTWEBENGINE_CHROMIUM_FLAGS']}
                phase('visible-rendering', [node, ROOT / 'scripts/verify_rendering_cdp.mjs', str(debug_port), out / 'visible'])
                phase('redundant-work', [node, ROOT / 'scripts/verify_redundant_work_cdp.mjs', str(debug_port), out / 'redundant-work', 'clean', '--keep-open'])
                phase('reader-responsiveness', [node, ROOT / 'scripts/verify_reader_responsiveness_cdp.mjs', str(debug_port), out / 'reader-responsiveness', 'final-exe:' + info['sha256']] + (['--geo-map'] if companion.is_dir() else []))
                phase('rendered-pixels', python + ['scripts/verify_rendering_pixels.py', out / 'visible'])
                from ctypes import wintypes
                from owned_window_capture import capture_owned_window
                report['nativeCapture'] = capture_owned_window(process.pid, out / 'native-owned.png')
                ctypes.windll.user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
                ctypes.windll.user32.PostMessageW(report['nativeCapture']['hwnd'], 0x0010, 0, 0)
                report['visibleExitCode'] = process.wait(timeout=60)
                if report['visibleExitCode'] != 0 or list(temp.glob('_MEI*')):
                    raise RuntimeError('Visible window failed normal exit/temp cleanup')
            finally:
                if process.poll() is None:
                    subprocess.run(['taskkill.exe', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL)
        report['success'] = True
        save()
        phase('cleanup-preview', python + ['scripts/cleanup_local_package.py', '--report', out / 'report.json'])
        phase('cleanup-apply', python + ['scripts/cleanup_local_package.py', '--report', out / 'report.json', '--apply'])
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        save()
    print(json.dumps({'success': report['success'], 'report': str(out / 'report.json')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
