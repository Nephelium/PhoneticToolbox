"""Build only the reviewed, local-only M03 runtime probe in a fresh output tree."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',type=Path);args=parser.parse_args();stage=args.stage.resolve()
    if not stage.is_relative_to(ROOT/'output/validation/m03-runtime') or not (stage/'payload/manifest.json').is_file():raise ValueError('Expected staged probe payload')
    if (stage/'build').exists() or (stage/'exe').exists():raise ValueError('Do not overwrite a previous probe build')
    subprocess.run([sys.executable,'-m','PyInstaller','--onefile','--console','--noupx','--name','M03-Runtime-Probe',
        '--distpath',str(stage/'exe'),'--workpath',str(stage/'build'),'--specpath',str(stage/'build'),
        '--paths',str(ROOT/'backend/src'),'--add-data',str(stage/'payload')+';payload',
        str(ROOT/'desktop/experiments/m03_runtime_probe.py')],cwd=ROOT,check=True)

if __name__=='__main__':main()
