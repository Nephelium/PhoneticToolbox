"""Owned browser acceptance for the Unt revision and reversible erhua decisions."""
import subprocess
import sys
from pathlib import Path

from verification_artifacts import verification_run

ROOT = Path(__file__).resolve().parents[1]


def main():
    with verification_run('m13-unt', 'chrome', 'Current Vue app; owned headless Chrome, Vite cache, Unt values, erhua, draft and PNG; no user corpus/database') as (out, scratch):
        result = subprocess.run(['node', str(ROOT / 'tests/e2e/m13-unt.cjs'), str(out), str(scratch), *sys.argv[1:]], cwd=ROOT,
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        (out / 'browser.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        print(result.stdout, end='')
        if result.returncode:
            print(result.stderr, end='')
            raise RuntimeError(f'Browser check failed: {out}')
        print(f'Evidence: {out}')


if __name__ == '__main__':
    main()
