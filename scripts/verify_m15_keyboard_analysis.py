"""Owned browser QA; artifacts/synthetic files are retired by the shared finalizer."""
import argparse
import os
from pathlib import Path
import subprocess
from verification_artifacts import verification_run

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', choices=['chrome', 'edge'], default='chrome')
    args = parser.parse_args()
    with verification_run('m15-keyboard-analysis', args.browser,
                          'Production AppShell; four synthetic PCM16 16kHz 0.2s WAV; independent browser profile; keyboard input, questionnaire, participant summary and export') as (out, scratch):
        env = dict(os.environ, M15_QA_OUT=str(out), M15_QA_SCRATCH=str(scratch), M15_QA_BROWSER=args.browser)
        result = subprocess.run(['node', str(ROOT / 'tests/e2e/m15-keyboard-analysis.cjs')], cwd=ROOT,
                                env=env, capture_output=True, text=True, encoding='utf8', errors='replace')
        (out / 'console.log').write_text(result.stdout + result.stderr, encoding='utf8')
        print(out, flush=True)
        if result.returncode:
            print(result.stderr, flush=True)
            raise RuntimeError(f'M15 {args.browser} QA failed; see report.json')
