"""Package the current workbench for the authorized local M12 EXE trial.

Uses the proven Research-Fix1 one-file recipe and its isolated m09-ui runtime.
Does not bundle the separate M03/M04 compatible runtime or change old EXEs.
"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NAME = 'PhoneticToolbox-v3-M12-R6'
if sys.platform != 'win32':
    raise SystemExit('This artifact is Windows-only')
if not (ROOT / 'frontend/dist/index.html').is_file():
    raise SystemExit('Build frontend first')
if (ROOT / 'dist/m12-preview-r6' / (NAME + '.exe')).exists():
    raise SystemExit('Existing M12 artifact retained; choose a new output name before rebuilding')
args = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--onefile', '--windowed',
        '--name', NAME, '--distpath', str(ROOT / 'dist/m12-preview-r6'),
        '--workpath', str(ROOT / 'output/build-m12-preview-r6'),
        '--specpath', str(ROOT / 'output/build-m12-preview-r6')]
for directory in ['scripts', 'backend/src', 'desktop/src', 'packages/phonetic_core/src']:
    args += ['--paths', str(ROOT / directory)]
for source, dest in [('frontend/dist', 'frontend/dist'), ('resources/vocal_tract', 'resources/vocal_tract'),
                     ('contracts', 'contracts'), ('backend/migrations', 'backend/migrations'),
                     ('docs/manual', 'docs/manual'), ('third_party/source-registry.json', 'third_party'),
                     ('third_party/licenses', 'third_party/licenses'),
                     ('phonetic_toolbox/core/acoustic/reaper.exe', 'resources/research')]:
    args += ['--add-data', str(ROOT / source) + ';' + dest]
for package in ['ptb_worker', 'ptb_api', 'phonetic_core', 'uvicorn']:
    args += ['--collect-submodules', package]
from m11_bundle import arguments as m11_bundle_arguments
args += m11_bundle_arguments(ROOT)
args += ['--collect-all', '_sounddevice_data', '--hidden-import', '_cffi_backend',
         '--exclude-module', 'matplotlib', '--exclude-module', 'IPython',
         str(ROOT / 'scripts/research_entry.py')]
env = os.environ.copy()
windows = Path(env.get('SystemRoot', 'C:/Windows'))
env['PATH'] = os.pathsep.join(str(p) for p in [Path(sys.executable).parent, Path(sys.base_prefix), windows / 'System32', windows])
subprocess.run(args, cwd=ROOT, env=env, check=True)
