"""Owned real-data M19 scale, interaction and fullscreen acceptance."""
import argparse
import os
from pathlib import Path
import subprocess
from verification_artifacts import verification_run

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--options-only', action='store_true')
    parser.add_argument('--native-options', action='store_true')
    args = parser.parse_args()
    with verification_run('geolinguistics-interaction', 'baseline' if args.baseline else 'chrome',
                          'Production build, real unchanged 16-file data, owned Chrome; live OL inspection and timed pan/hover; no geography audit') as (out, scratch):
        print(out, flush=True)
        result = subprocess.run(['node', str(ROOT/'tests/e2e/geolinguistics-interaction.cjs'), str(out), str(scratch),
                                 str(args.source.resolve()), 'baseline' if args.baseline else 'native-options' if args.native_options else 'options' if args.options_only else 'acceptance'],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        (out/'browser.log').write_text(result.stdout+result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError('M19 interaction acceptance failed: '+str(out/'report.json'))
