"""M13-ACADEMIC-SOURCES: owned browser profiles, source UI and copied citations."""
import subprocess
from pathlib import Path

from verification_artifacts import verification_run

ROOT = Path(__file__).resolve().parents[1]


def main():
    with verification_run(
        'm13-sources', 'chrome',
        'Current Vue app; isolated headless Chrome profile and Vite cache; '
        'M13 and About citations, light/dark at 1440/900 px; no research input or database',
    ) as (out, scratch):
        result = subprocess.run(
            ['node', str(ROOT / 'tests/e2e/m13-sources.cjs'), str(out), str(scratch)],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
            errors='replace', timeout=120, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        (out / 'browser.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        print(result.stdout, end='')
        if result.returncode:
            print(result.stderr, end='')
            raise RuntimeError(f'Browser check failed: {out}')
        print(f'Evidence: {out}')


if __name__ == '__main__':
    main()
