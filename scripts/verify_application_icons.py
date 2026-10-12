"""Keep icon preference test inputs in the owned verification scratch lifecycle."""
import os
from pathlib import Path
import subprocess
import sys

from verification_artifacts import verification_run

ROOT = Path(__file__).resolve().parents[1]

with verification_run('settings-icons', 'native', '31 actual artwork renders; selection reopening; canonical import, limits and atomic failure preservation') as (out, scratch):
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'desktop/src'), QT_QPA_PLATFORM='offscreen')
    result = subprocess.run([sys.executable, '-m', 'pytest', '-o', 'addopts=', 'desktop/tests/test_application_icons.py', 'desktop/tests/test_p19_icon.py', '--basetemp', str(scratch / 'pytest'), '-q'], cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
    (out / 'pytest.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    print(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('Native icon verification failed')
