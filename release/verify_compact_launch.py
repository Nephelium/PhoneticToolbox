"""Run the actual EXE outside the checkout with owned state and no dev bindings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario', choices=('distribution', 'textgrid'), default='distribution')
    options = parser.parse_args()
    source, output = options.package.resolve(), options.output.resolve()
    if not source.is_relative_to(ROOT / 'dist') or not output.is_relative_to(Path('D:/PTB-Compact-QA-20261006')):
        raise ValueError('Expected owned build and new D-drive QA directory')
    output.mkdir(parents=True, exist_ok=False)
    executable = output / 'PhoneticToolbox.exe'
    shutil.copyfile(source / 'PhoneticToolbox.exe', executable)
    with executable.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    descriptor = json.loads((source / 'application.json').read_text('utf8'))
    if sha != descriptor['executable']['sha256'] or executable.stat().st_size > 500_000_000:
        raise ValueError('Relocated EXE identity mismatch')
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('PTB_', 'PYTHON', 'CONDA', '_PYI')) and key not in (
                       'VIRTUAL_ENV', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QTWEBENGINEPROCESS_PATH',
                       'QTWEBENGINE_RESOURCES_PATH', 'QTWEBENGINE_LOCALES_PATH')}
    profile, temp = output / 'profile', output / 'temp'
    profile.mkdir(); temp.mkdir()
    environment.update(LOCALAPPDATA=str(profile), TEMP=str(temp), TMP=str(temp),
                       PTB_OWNED_BOOTSTRAP_LOG=str(output/'bootstrap-error.log'),
                       PATH=os.pathsep.join((os.environ['SystemRoot'] + '/System32', os.environ['SystemRoot'])),
                       QT_QPA_PLATFORM='windows',
                       QTWEBENGINE_CHROMIUM_FLAGS='--mute-audio --autoplay-policy=no-user-gesture-required')
    started = time.monotonic()
    with (output / 'process.log').open('wb') as log:
        flag = '--verify-distribution' if options.scenario == 'distribution' else '--verify-m12'
        process = subprocess.Popen([str(executable), flag, str(output / options.scenario)],
                                   cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        code = process.wait(timeout=900)
    evidence = json.loads((output / options.scenario / 'report.json').read_text('utf8'))
    report = dict(success=code == 0 and evidence.get('success') is True, scenario=options.scenario, exitCode=code, seconds=time.monotonic() - started,
                  exeSha256=sha, developerEnvironmentRemoved=True, relocatedExe=str(executable),
                  leftoverBootDirectories=[str(path) for path in temp.glob('_MEI*')],
                  scope='Actual relocated frozen EXE, isolated owned data and Windows-only PATH; not a separate physical computer or physical device test')
    (output / 'launch-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf8')
    print(json.dumps(report), flush=True)
    if not report['success'] or report['leftoverBootDirectories']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
