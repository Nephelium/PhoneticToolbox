"""Build the authorized local Windows single-file recording artifact."""
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if sys.platform!='win32':raise SystemExit('This artifact is Windows-only')
if not (ROOT/'frontend/dist/index.html').exists():raise SystemExit('Build frontend first')
args=[sys.executable,'-m','PyInstaller','--noconfirm','--onefile','--windowed',
      '--name','PhoneticToolbox-v3-M10-R5','--distpath',str(ROOT/'dist/m10-recording'),
      '--workpath',str(ROOT/'output/build-m10'),'--specpath',str(ROOT/'output/build-m10'),
      '--paths',str(ROOT/'scripts'),
      '--add-data',str(ROOT/'frontend/dist')+';frontend/dist',
      '--add-data',str(ROOT/'resources/vocal_tract')+';resources/vocal_tract',
      '--add-data',str(ROOT/'contracts')+';contracts',
      '--add-data',str(ROOT/'docs/manual/vocal-tract.md')+';docs/manual',
      '--add-data',str(ROOT/'scripts/build_m10_geometry.py')+';scripts',
      '--add-data',str(ROOT/'third_party/source-registry.json')+';third_party',
      '--add-data',str(ROOT/'third_party/licenses')+';third_party/licenses',
      '--collect-all','_sounddevice_data','--hidden-import','_cffi_backend',
      '--collect-submodules','uvicorn',
      '--exclude-module','matplotlib','--exclude-module','IPython',
      str(ROOT/'scripts/m10_recording_entry.py')]
if '--incremental' not in sys.argv:args.insert(3,'--clean')
# Do not harvest native runtimes from the launching editor's tools on PATH.
# The wheel, Python runtime and Windows provide this artifact's dependencies.
env=os.environ.copy()
windows=Path(env.get('SystemRoot','C:/Windows'))
env['PATH']=os.pathsep.join(str(p) for p in [Path(sys.executable).parent,Path(sys.base_prefix),windows/'System32',windows])
subprocess.run(args,cwd=ROOT,env=env,check=True)
