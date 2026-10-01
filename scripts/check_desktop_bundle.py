"""Read-only preflight for a staged portable desktop bundle, on its build platform."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'desktop/src'))
    from ptb_desktop.bundle_manifest import runtime_bindings
    result=runtime_bindings(args.bundle)
    print(json.dumps({'runtime_bindings_valid':True,'runtimes':sorted(result),
        'native_functionality_verified':False,'release_ready':False}))


if __name__=='__main__':main()
