"""Run preference lifecycle checks in an independent owned Chrome profile."""
from pathlib import Path
import subprocess
from verification_artifacts import verification_run

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    with verification_run('web-settings', 'browser', 'Built Vue UI, synthetic HTTP, settings refresh/logout/new-workspace/account boundaries') as (out, scratch):
        with (out / 'build.log').open('w', encoding='utf-8') as log:
            subprocess.run(['node', str(ROOT / 'frontend/node_modules/vite/bin/vite.js'), 'build', '--outDir', str(scratch / 'frontend')], cwd=ROOT / 'frontend', stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['node', str(ROOT / 'tests/e2e/web-settings.cjs'), str(out), str(scratch)], cwd=ROOT, check=True)
