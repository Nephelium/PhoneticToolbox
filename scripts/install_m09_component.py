"""Install the pinned local component ZIP without publishing or downloading."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend/src'),str(ROOT/'packages/phonetic_core/src')]
from ptb_worker.vocoder_components import install

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True)
    print(json.dumps(install(parser.parse_args().archive),ensure_ascii=False,indent=2))
