"""Detached direct launch of an EXE alone, with owned state and Windows PATH."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    source, output = args.package.resolve(), args.output.resolve()
    if not source.is_relative_to(ROOT / 'dist') or not output.is_relative_to(Path('D:/PTB-Compact-QA-20261006')):
        raise ValueError('Expected owned compact build and new D-drive QA directory')
    with socket.socket() as check:
        check.bind(('127.0.0.1', args.port))
    output.mkdir(parents=True, exist_ok=False)
    payload = output / 'payload'; payload.mkdir()
    executable = payload / 'PhoneticToolbox.exe'
    shutil.copyfile(source / executable.name, executable)
    profile, temp = output / 'profile', output / 'temp'
    profile.mkdir(); temp.mkdir()
    updates = profile / 'PhoneticToolbox/v3/updates'; updates.mkdir(parents=True)
    (updates / 'state.json').write_text('{"autoCheck":false}', 'utf8')
    marker = profile / 'PhoneticToolbox/v3/project-do-not-remove.json'
    marker.write_text('{"fixture":"owned direct launch project"}', 'utf8')
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('PTB_', 'PYTHON', 'CONDA', '_PYI')) and key not in (
                       'VIRTUAL_ENV', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QTWEBENGINEPROCESS_PATH',
                       'QTWEBENGINE_RESOURCES_PATH', 'QTWEBENGINE_LOCALES_PATH')}
    environment.update(LOCALAPPDATA=str(profile), TEMP=str(temp), TMP=str(temp),
                       PATH=os.pathsep.join((os.environ['SystemRoot'] + '/System32', os.environ['SystemRoot'])),
                       QT_QPA_PLATFORM='offscreen', QTWEBENGINE_REMOTE_DEBUGGING=str(args.port),
                       QTWEBENGINE_CHROMIUM_FLAGS='--mute-audio --disable-gpu --autoplay-policy=no-user-gesture-required')
    process = subprocess.Popen([str(executable)], cwd=payload, env=environment, close_fds=True,
                               creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    launch = output / 'launch'; launch.mkdir()
    request = launch / 'request.json'; request.write_text(json.dumps(dict(pid=process.pid)), 'utf8')
    (launch / 'status.json').write_text(json.dumps(dict(state='started', pid=process.pid)), 'utf8')
    plan = dict(scope='Actual EXE alone outside checkout, detached without console or redirected IO, Windows-only PATH, fresh owned profile; not a physical-device or separate-computer test',
                kind='portable-onefile', launchMode='direct', request=str(request), origin=str(executable),
                output=str(output), profile=str(profile), port=args.port, preferences={}, seedFresh=True,
                project_marker=str(marker), project_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(), closeAfter=True)
    plan_file = output / 'handoff.json'; plan_file.write_text(json.dumps(plan), 'utf8')
    subprocess.run(['C:/Program Files/nodejs/node.exe', str(ROOT / 'scripts/verify_update_relaunch.mjs'), str(plan_file)], check=True)
    code = process.wait(timeout=60)
    report = dict(success=code == 0, exitCode=code, noInstaller=True, exeOnly=True,
                  developerEnvironmentRemoved=True, leftoverBootDirectories=[str(path) for path in temp.glob('_MEI*')])
    (output / 'direct-report.json').write_text(json.dumps(report, indent=2), 'utf8')
    print(json.dumps(report), flush=True)
    if code != 0 or report['leftoverBootDirectories']:
        raise RuntimeError('Direct GUI did not exit cleanly')


if __name__ == '__main__':
    main()
