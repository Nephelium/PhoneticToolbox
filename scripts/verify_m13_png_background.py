"""Production PNG background acceptance in an isolated headless browser."""
from pathlib import Path
import subprocess
from verification_artifacts import verification_run
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    with verification_run('m13-png-background','chrome','Owned headless Chrome; synthetic split/solid PNG, current/custom background/effects, decoded output pixels and preview/download bytes') as (out,scratch):
        result=subprocess.run(['node',str(ROOT/'tests/e2e/m13-png-background.cjs'),str(out),str(scratch)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240,creationflags=subprocess.CREATE_NO_WINDOW)
        (out/'browser.log').write_text(result.stdout+result.stderr,encoding='utf-8');print(out,flush=True)
        if result.returncode:print(result.stderr,flush=True);raise RuntimeError('M13 PNG background browser acceptance failed')
