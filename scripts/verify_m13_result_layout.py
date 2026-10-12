"""Owned real-page geometry and PNG acceptance for M13 result layout."""
import subprocess
from pathlib import Path
from verification_artifacts import verification_run

ROOT=Path(__file__).resolve().parents[1]
with verification_run('m13-result-layout','chrome','Owned Chrome and Vite; text layout, draft and PNG, no scientific jobs or user database') as (out,scratch):
    result=subprocess.run(['node',str(ROOT/'tests/e2e/m13-result-layout.cjs'),str(out),str(scratch)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
    (out/'browser.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    print(result.stdout,end='')
    if result.returncode:
        print(result.stderr,end='')
        raise RuntimeError(str(out))
    print('Evidence:',out)
