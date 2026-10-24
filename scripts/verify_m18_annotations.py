"""Run selection regression with an owned Chrome profile and scratch cleanup."""
from pathlib import Path
import subprocess
from verification_artifacts import verification_run

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    with verification_run('m18', 'annotations-browser',
                          'Shared PdfReader; synthetic two-page text; actual pointer/keyboard controls; owned Chrome') as (out, scratch):
        print(out, flush=True)
        subprocess.run(['node', str(ROOT / 'tests/e2e/m18-annotations.cjs'), str(out), str(scratch)],
                       cwd=ROOT, check=True)
