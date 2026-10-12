"""Owned browser verification of normalized global and module citations."""
import subprocess
from pathlib import Path
from verification_artifacts import verification_run
ROOT=Path(__file__).resolve().parents[1]
def main():
    with verification_run('source-citation-normalization','chrome','Read-only citation UI, private-CIN hiding and clipboard capture; no research inputs or database') as (out,scratch):
        result=subprocess.run(['node',str(ROOT/'tests/e2e/source-citations.cjs'),str(out),str(scratch)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
        (out/'browser.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        print(result.stdout,end='')
        if result.returncode:
            print(result.stderr,end='')
            raise RuntimeError(f'Citation UI failed: {out}')
        print(f'Evidence: {out}')
if __name__=='__main__':main()
