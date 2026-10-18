"""Refresh M10 resource identities without overwriting historical source attribution."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDERS=('frontend/public/vocal-tract','resources/vocal_tract')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def encode(value):return json.dumps(value,ensure_ascii=False,indent=2)+'\n'
def read(path):return json.loads((ROOT/path).read_text('utf-8'))

def main(check=False):
    paths=sorted(p for folder in FOLDERS for p in (ROOT/folder).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    bundle=read('resources/vocal_tract/bundle-manifest.json')
    bundle['version']='3.0.0-m10.r21'
    bundle['files']=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p))
                     for p in paths if p.name!='bundle-manifest.json']
    updates={'resources/vocal_tract/bundle-manifest.json':encode(bundle)}
    manifest=read('contracts/resource-manifest.json')
    manifest['resources']=[x for x in manifest['resources'] if not x['path'].startswith(FOLDERS)]
    for p in paths:
        relative=p.relative_to(ROOT).as_posix()
        sid='SRC-VTL' if relative.startswith('resources/') else 'SRC-THREE' if '/vendor/' in relative else 'ASSET-HEAD' if p.name=='head.json' else 'ASSET-NASAL' if p.name=='nasal.json' else 'PROJECT-M10'
        sha=hashlib.sha256(updates[relative].encode()).hexdigest() if relative in updates else digest(p)
        manifest['resources'].append(dict(path=relative,sha256=sha,source_id=sid))
    updates['contracts/resource-manifest.json']=encode(manifest)
    evidence=read('docs/modules/evidence/M10-migration.json')
    for item in evidence['files']:
        target=ROOT/item['target']
        if target.is_file():item['current_sha256']=hashlib.sha256(updates[item['target']].encode()).hexdigest() if item['target'] in updates else digest(target)
    evidence['behavior_revision']='R21 smoothing and source features; Windows/Wasm rebuilt, Mac R21 build pending; see docs/testing/m10-smoothing-review.md'
    updates['docs/modules/evidence/M10-migration.json']=encode(evidence)
    changed=[]
    for name,content in updates.items():
        p=ROOT/name
        if p.read_text('utf-8')!=content:
            changed.append(name)
            if not check:p.write_text(content,encoding='utf-8',newline='\n')
    if check and changed:raise SystemExit('Stale M10 resources: '+', '.join(changed))
    print(json.dumps({'resources':len(paths),'changed':changed,'check':check}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    main(parser.parse_args().check)
