"""Build the independent one-file inbox with only its shared UI graph."""
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
    web=ROOT/'output/build-feedback-manager/web'
    if not (web/'feedback-manager.html').is_file():raise SystemExit('Build the shared inbox frontend first')
    # Foreign ICU libraries on the caller's PATH can override Windows' ICU API.
    # Qt imports unversioned ICU exports supplied by supported Windows systems.
    env={**os.environ,'PATH':os.pathsep.join([os.environ['SystemRoot']+'/System32',os.environ['SystemRoot']])}
    subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm','--onefile','--windowed','--name','反馈收件箱',
        '--copy-metadata','ptb-desktop','--paths',str(ROOT/'desktop/src'),'--add-data',str(web)+';frontend',
        '--icon',str(web/'favicon.ico'),
        '--distpath',str(ROOT/'output/feedback-manager/1.0.1'),
        '--workpath',str(ROOT/'output/build-feedback-manager/pyinstaller-windows'),
        '--specpath',str(ROOT/'output/build-feedback-manager'),
        str(ROOT/'tools/feedback-manager/windows/entry.py')],check=True,cwd=ROOT,env=env)
