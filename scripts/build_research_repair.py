"""Build a separate, local-only Windows one-file repair candidate. Keep R5."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
args = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--onefile', '--windowed',
        '--name', 'PhoneticToolbox-v3-Research-Fix1',
        '--distpath', str(ROOT / 'dist/research-repair'),
        '--workpath', str(ROOT / 'output/build-research-repair'),
        '--specpath', str(ROOT / 'output/build-research-repair')]
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
args += ['--collect-all', '_sounddevice_data', '--hidden-import', '_cffi_backend',
         '--exclude-module', 'matplotlib', '--exclude-module', 'IPython',
         str(ROOT / 'scripts/research_entry.py')]
env = os.environ.copy()
windows = Path(env.get('SystemRoot', 'C:/Windows'))
env['PATH'] = os.pathsep.join(str(p) for p in [Path(sys.executable).parent, Path(sys.base_prefix), windows / 'System32', windows])
subprocess.run(args, cwd=ROOT, env=env, check=True)
