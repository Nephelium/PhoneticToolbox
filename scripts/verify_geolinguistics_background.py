"""Visible final-EXE layered-map regression and independent screenshot pixels."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

from PIL import Image
from owned_window_capture import capture_owned_window
from portable_cache_qa import seed_portable_profile
from verification_artifacts import verification_run, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--unpacked-exe', type=Path, help='Diagnostic only; default always launches the final distribution EXE')
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    info = json.loads((package/'build-info.json').read_text('utf-8'))
    distribution = package/info['exe']
    with distribution.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == info['sha256']
    assert (package/'geolinguistics-data/data-manifest.json').is_file()
    executable = args.unpacked_exe.resolve(strict=True) if args.unpacked_exe else distribution
    with verification_run('geolinguistics-background', 'exe', 'Visible Windows Qt/default GPU; isolated profile; 12 appearance/count cases, 48 gestures, real final map data, actual screenshot land pixels') as (out, scratch):
        print(out, flush=True)
        profile, temp = scratch/'profile', scratch/'temp'
        profile.mkdir(); temp.mkdir(); seed_portable_profile(profile)
        for folder, name, value in [('updates','state.json',{'autoCheck':False}), ('papers','preferences.json',{'autoReceive':False,'checkIntervalHours':6})]:
            target = profile/'PhoneticToolbox/v3'/folder
            target.mkdir(parents=True, exist_ok=True)
            write_json(target/name, value)
        with socket.socket() as available:
            available.bind(('127.0.0.1', 0)); port = available.getsockname()[1]
        env = {k:v for k,v in os.environ.items() if not k.startswith(('PTB_','PYTHON','CONDA','_PYI','QT_','QTWEBENGINE')) and k!='VIRTUAL_ENV'}
        env.update(LOCALAPPDATA=str(profile), TEMP=str(temp), TMP=str(temp),
                   PATH=os.pathsep.join([os.environ['SystemRoot']+'/System32',os.environ['SystemRoot']]),
                   QT_QPA_PLATFORM='windows', QTWEBENGINE_REMOTE_DEBUGGING=str(port),
                   QTWEBENGINE_CHROMIUM_FLAGS='--mute-audio --remote-allow-origins=*')
        if args.unpacked_exe:
            env['PTB_DISTRIBUTION_EXE'] = str(distribution)
        write_json(out/'identity.json', dict(exe=str(executable), distribution=str(distribution), sha256=info['sha256'], diagnosticUnpacked=bool(args.unpacked_exe)))
        with (out/'process.log').open('wb') as log:
            process = subprocess.Popen([str(executable)], cwd=scratch, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                result = subprocess.run(['node',str(ROOT/'tests/e2e/geolinguistics-background.cjs'),str(port),str(out)], timeout=540)
                captured = capture_owned_window(process.pid, out/'native-window.png')
                ctypes.windll.user32.PostMessageW.argtypes = [wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
                ctypes.windll.user32.PostMessageW(captured['hwnd'],0x0010,0,0)
                exit_code = process.wait(timeout=60)
                write_json(out/'exit.json', dict(testCode=result.returncode, exeCode=exit_code, nativeCapture=captured))
                if result.returncode or exit_code:
                    raise RuntimeError('Final EXE background regression failed')
                report = json.loads((out/'background-report.json').read_text('utf-8'))
                pixel_rows = []
                for case in report['cases']:
                    for state in case['states']:
                        if 'screenshot' not in state:
                            continue
                        with Image.open(out/state['screenshot']) as image:
                            image = image.convert('RGB')
                            for sample in state['samples']:
                                x,y = sample['x'],sample['y']
                                assert 0<=x<image.width and 0<=y<image.height
                                expected = tuple(sample['rgb'])
                                matches = any(image.getpixel((a,b))==expected for a in range(max(0,x-1),min(image.width,x+2)) for b in range(max(0,y-1),min(image.height,y+2)))
                                assert matches, (state['screenshot'], x, y, image.getpixel((x,y)), expected)
                        pixel_rows.append(dict(screenshot=state['screenshot'],landSamples=len(state['samples'])))
                write_json(out/'pixels.json', dict(success=True,cases=pixel_rows,scope='Actual EXE screenshot RGB vs opaque land pixels, avoiding point overlays; 1px raster alignment neighbourhood'))
            finally:
                if process.poll() is None:
                    subprocess.run(['taskkill.exe','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
