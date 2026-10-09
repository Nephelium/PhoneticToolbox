"""Prepare a reviewed, licensed PDF pair and a complete static catalogue.

No network or credentials. Does not touch the application resource directories.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'desktop/src'))
from ptb_desktop.papers import validate_catalog


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--translation',type=Path,required=True)
    parser.add_argument('--catalog',type=Path,help='Existing full catalogue; preserve older entries.')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    metadata=json.loads(args.metadata.read_text('utf8'))
    catalog=json.loads(args.catalog.read_text('utf8')) if args.catalog else {'schema':'ptb-papers/1','papers':[]}
    files=[]
    for kind in ('original','translation'):
        path=getattr(args,kind);raw=path.read_bytes()
        if not raw.startswith(b'%PDF-'):raise ValueError('Expected PDF: '+str(path))
        digest=hashlib.sha256(raw).hexdigest()
        metadata[kind]={'path':f'{metadata["id"]}/{kind}-{digest[:16]}.pdf','size':len(raw),'sha256':digest}
        files.append((path,metadata[kind]['path']))
    catalog['papers']=[p for p in catalog['papers'] if p['id']!=metadata['id']]+[metadata]
    validate_catalog(catalog)
    for source,relative in files:
        target=args.output/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    (args.output/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps({'papers':len(catalog['papers']),'output':str(args.output.resolve())}))


if __name__=='__main__':main()
