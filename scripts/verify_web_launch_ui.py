"""Built web UI in independent Chrome; HTTP transport is explicitly synthetic."""
from pathlib import Path
import subprocess
import argparse
from verification_artifacts import verification_run

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--browser',choices=['chrome','edge'],default='chrome');args=parser.parse_args()
    with verification_run('web-launch','browser','Built Vue UI, synthetic HTTP accounts/workspaces; independent visible '+args.browser+', light/dark, account isolation') as (out,scratch):
        result=subprocess.run(['node',str(ROOT/'tests/e2e/web-launch.cjs'),str(out),str(scratch),args.browser],cwd=ROOT,check=False)
        if result.returncode:raise SystemExit(result.returncode)
