"""Simulated catalogue combinations in an owned standalone Chrome profile."""
from pathlib import Path
import subprocess
from verification_artifacts import verification_run
ROOT = Path(__file__).resolve().parents[1]
with verification_run('m18', 'r3-browser', 'Chrome isolated profile; simulated multi-year catalogue; actual Vue archive selection and preference UI') as (out, scratch):
    subprocess.run(['node', str(ROOT/'tests/e2e/m18-r3.cjs'), str(out), str(scratch)], cwd=ROOT, check=True)
