"""Owned native geometry evidence; no audio device or user profile access."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'packages/phonetic_core/src'))
from phonetic_core.vocal_tract.engine import Engine


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',action='store_true');args=parser.parse_args()
    out=ROOT/'output/validation/m10-r11';out.mkdir(parents=True,exist_ok=True)
    if args.baseline:
        backup=out/'before';backup.mkdir(exist_ok=True)
        paths=list((ROOT/'frontend/public/vocal-tract').glob('*.*'))+list((ROOT/'packages/phonetic_core/src/phonetic_core/vocal_tract').glob('*.py'))+list((ROOT/'desktop/src/ptb_desktop/vocal_tract').glob('*.py'))+list((ROOT/'resources/vocal_tract/sources').glob('*.*'))+list((ROOT/'resources/vocal_tract/native').glob('*.*'))+[ROOT/'scripts/build_m10_geometry.py',ROOT/'third_party/source-registry.json']
        hashes={}
        for src in paths:
            rel=src.relative_to(ROOT);dest=backup/rel
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
            hashes[str(rel)]=hashlib.sha256(dest.read_bytes()).hexdigest()
        (backup/'hashes.json').write_text(json.dumps(hashes,indent=2),'utf-8')
    engine=Engine(resource_dir=ROOT/'resources/vocal_tract/native');cases=[]
    try:
        for name in ['a','i','u','n']:
            for vo in [0,.25,.5,.75,1.,1.5]:
                p=engine.presets[name].copy();p[engine.names.index('VO')]=vo
                cases.append({'name':f'{name}-VO-{vo}','state':engine.snapshot(p)})
        for x in [4.4,4.6,4.8,5.,5.5]:
            p=engine.presets['a'].copy();p[engine.names.index('TTX')]=x
            cases.append({'name':f'tip-{x}','state':engine.snapshot(p)})
        print(json.dumps({'meta':engine.metadata(),'a_contours':cases[0]['state']['contours'],'tip_readback':[{ 'name':c['name'],'limited':c['state']['limited'][10:14]} for c in cases if c['name'].startswith('tip')]},ensure_ascii=False))
        (out/('baseline.json' if args.baseline else 'current.json')).write_text(json.dumps(cases),'utf-8')
    finally:engine.close()


if __name__=='__main__':main()
