"""Actual Vue controls with fixture transfers in an isolated owned Chrome."""
from pathlib import Path
import subprocess
from verification_artifacts import verification_run

ROOT=Path(__file__).resolve().parents[1]
with verification_run('m11-delivery','browser','Simulated transport; real Vue, download speed/bytes/progress, ZIP/folder actions, themes') as (out,scratch):
    subprocess.run(['node',str(ROOT/'tests/e2e/m11-delivery.cjs'),str(out),str(scratch)],cwd=ROOT,check=True)
