"""Launch the actual self-extracted GUI without console or redirected IO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    temp = Path(os.environ['TEMP']).resolve()
    payload = args.payload.resolve()
    assert payload.is_relative_to(temp) and payload.name.startswith('PTB-Preview1-')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = temp / ('PTB-Preview1-20261005-selfextract-launch-'+uuid4().hex)
    profile.mkdir()
    storage = profile / 'PhoneticToolbox-v3/workbench'
    updates = profile / 'PhoneticToolbox/v3/updates'
    updates.mkdir(parents=True)
    (updates / 'state.json').write_text('{"autoCheck":false}', 'utf8')
    marker = profile / 'PhoneticToolbox/v3/project-do-not-remove.json'
    marker.write_text('{"fixture":"owned self-extractor QA project"}', 'utf8')
    with socket.socket() as check:
        check.bind(('127.0.0.1', args.port))
    environment = dict(os.environ, LOCALAPPDATA=str(profile), QT_QPA_PLATFORM='offscreen',
                       QTWEBENGINE_REMOTE_DEBUGGING=str(args.port),
                       QTWEBENGINE_CHROMIUM_FLAGS='--mute-audio --disable-gpu')
    exe = payload / 'PhoneticToolbox.exe'
    process = subprocess.Popen([str(exe)], env=environment, close_fds=True,
                               creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    request = output / 'launch/request.json'
    request.parent.mkdir()
    request.write_text(json.dumps({'exe': str(exe), 'pid': process.pid, 'launchMode': 'direct-no-console'}), 'utf8')
    (request.parent / 'status.json').write_text(json.dumps({'state': 'started', 'pid': process.pid}), 'utf8')
    plan = dict(scope='Actual final self-extractor payload, detached GUI without console or IO redirects, fresh owned QA data; no migration or real user data claim',
                kind='portable-selfextract', launchMode='direct', request=str(request), origin=str(exe),
                output=str(output), profile=str(profile), storage=str(storage), port=args.port,
                preferences={}, seedFresh=True, project_marker=str(marker),
                project_sha256=hashlib.sha256(marker.read_bytes()).hexdigest())
    (output / 'handoff.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n', 'utf8')
    print(json.dumps({'plan': str(output / 'handoff.json'), 'pid': process.pid}), flush=True)


if __name__ == '__main__':
    main()
