"""Isolated real browser and synthetic in-memory files, owned profile lifecycle."""
import os
from pathlib import Path
import subprocess
from verification_artifacts import verification_run

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    with verification_run('m12','three-tiers-browser','Production AppShell; in-memory 3s mono PCM16 16kHz sine WAV; three interval tiers and a preserved point tier; drag/input/resize and canvas activity probes') as (out,scratch):
        print(out,flush=True)
        result=subprocess.run(['node',str(ROOT/'tests/e2e/m12-three-tiers.cjs')],cwd=ROOT,
                              env=dict(os.environ,M12_QA_OUT=str(out),M12_QA_SCRATCH=str(scratch)),
                              capture_output=True,text=True,encoding='utf-8',errors='replace')
        (out/'console.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        print(result.stdout+result.stderr,flush=True)
        if result.returncode:raise RuntimeError('M12 browser verification failed; see report.json')
