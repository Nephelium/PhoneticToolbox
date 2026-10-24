"""Run the manual opening-link and website hash regressions in owned scratch."""
import argparse
from pathlib import Path
import subprocess
from verification_artifacts import verification_run
from workbench_source import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,default=ROOT/'output/validation/manual-tutorial-links-20261024/before.json')
    args=parser.parse_args()
    node='C:/Users/13680/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
    with verification_run('manual-tutorial-links','focused','Read-only 22 chapter opening links and exact pre-edit body/search comparison; owned static website/browser; inert external player, no video/network playback or publishing') as (out,scratch):
        print(out,flush=True)
        with (out/'browser.log').open('w',encoding='utf8') as log:
            subprocess.run([node,str(ROOT/'tests/e2e/manual-tutorial-links.cjs'),str(out),str(scratch),str(args.before.resolve())],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
