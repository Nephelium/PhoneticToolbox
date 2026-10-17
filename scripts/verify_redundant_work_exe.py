"""Run only the requested EXE with isolated data, then close its owned process tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

from pypdf import PdfWriter
from verification_artifacts import verification_run, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--expect-clean', action='store_true')
    parser.add_argument('--native-capture', action='store_true')
    args = parser.parse_args()
    exe = args.exe.resolve(strict=True)
    with exe.open('rb') as stream:
        identity = dict(path=str(exe), bytes=exe.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
    root = Path(__file__).resolve().parents[1]
    with verification_run('redundant-work', 'exe', 'Actual specified EXE, generated four-page PDF, own LOCALAPPDATA/TEMP, visible Windows/default GPU, twelve M18/home switches') as (out, scratch):
        profile, temp, library = scratch/'profile', scratch/'temp', scratch/'library'
        for folder in (profile, temp, library):
            folder.mkdir()
        writer = PdfWriter()
        for _ in range(4):
            writer.add_blank_page(width=612, height=792)
        with (library/'Owned performance fixture.pdf').open('wb') as stream:
            writer.write(stream)
        data = profile/'PhoneticToolbox/v3'
        for folder in ('papers', 'updates'):
            (data/folder).mkdir(parents=True)
        write_json(data/'papers/preferences.json', dict(autoReceive=False, libraryRoot=str(library), checkIntervalHours=6))
        write_json(data/'updates/state.json', dict(autoCheck=False))
        with socket.socket() as available:
            available.bind(('127.0.0.1', 0))
            port = available.getsockname()[1]
        env = {key:value for key,value in os.environ.items() if not key.startswith(('PTB_', 'PYTHON', 'CONDA', '_PYI', 'QT_')) and key != 'VIRTUAL_ENV'}
        env.update(LOCALAPPDATA=str(profile), TEMP=str(temp), TMP=str(temp),
                   PATH=os.pathsep.join((os.environ['SystemRoot']+'/System32', os.environ['SystemRoot'])),
                   QT_QPA_PLATFORM='windows', QTWEBENGINE_REMOTE_DEBUGGING=str(port),
                   QTWEBENGINE_CHROMIUM_FLAGS='--mute-audio --remote-allow-origins=*', PTB_OWNED_BOOTSTRAP_LOG=str(out/'bootstrap.log'))
        write_json(out/'identity.json', identity)
        with (out/'process.log').open('wb') as log:
            process = subprocess.Popen([str(exe)], cwd=scratch, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                result = subprocess.run(['node', str(root/'scripts/verify_redundant_work_cdp.mjs'), str(port), str(out), 'clean' if args.expect_clean else 'baseline', *(['--keep-open'] if args.native_capture else [])], cwd=root, timeout=150, capture_output=True, text=True, encoding='utf-8')
                (out/'runner.log').write_text(result.stdout+result.stderr, encoding='utf-8')
                if result.returncode:
                    raise RuntimeError(result.stdout+result.stderr)
                if args.native_capture:
                    from owned_window_capture import capture_owned_window
                    import ctypes
                    from ctypes import wintypes
                    native = capture_owned_window(process.pid, out/'native-owned.png')
                    write_json(out/'native-capture.json', native)
                    ctypes.windll.user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
                    ctypes.windll.user32.PostMessageW(native['hwnd'], 0x0010, 0, 0)
            finally:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill.exe', '/PID', str(process.pid), '/T', '/F'], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    process.wait(timeout=10)
            report = json.loads((out/'cdp-report.json').read_text('utf-8'))
            write_json(out/'report.json', dict(success=report['success'], identity=identity, exit_code=process.returncode, repeated_renders=len(report.get('switches', []))))
            print(json.dumps(dict(out=str(out), identity=identity, repeated_renders=len(report.get('switches', [])), success=report['success'])), flush=True)


if __name__ == '__main__':
    main()
