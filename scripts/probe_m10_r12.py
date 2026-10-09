"""Capture owned native poses for M10-R12 geometry and UI verification."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packages/phonetic_core/src'))
from phonetic_core.vocal_tract.engine import Engine


def main():
    out=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'output/validation/m10-r12';out.mkdir(parents=True,exist_ok=True)
    e=Engine(resource_dir=ROOT/'resources/vocal_tract/native');cases=[];summary=[]
    try:
        for name in e.presets:
            cases.append({'name':name,'state':e.snapshot(e.presets[name])})
        for label,targets in [('blade',np.linspace(-.575,-2.5,31)),('retract',np.linspace(4.4,1.5,31))]:
            for target in targets:
                p=e.presets['a'].copy()
                if label=='blade':p[13]=target
                else:p[8]=.0524;p[9]=-2.1;p[10]=target;p[11]=.25;p[12]=2.6;p[13]=-1.3
                s=e.snapshot(p);cases.append({'name':f'{label}-{target:.3f}','state':s})
                summary.append({'case':cases[-1]['name'],'requested':list(p[10:14]),'limited':s['limited'][10:14],'min_area':min(s['tube_areas'])})
        for tx in [1.5,2,2.5,3]:
            for ty in [-.5,.25,.8]:
                for by in [-1,-1.5,-2]:
                    p=e.presets['a'].copy();p[9]=-2.1;p[10]=tx;p[11]=ty;p[12]=2.6;p[13]=by
                    cases.append({'name':f'grid-{tx}-{ty}-{by}','state':e.snapshot(p)})
        (out/'current.json').write_text(json.dumps(cases),'utf-8')
        (out/'control-scan.json').write_text(json.dumps(summary,indent=2),'utf-8')
        print(json.dumps({'poses':len(cases),'first_blade':summary[0],'last_blade':summary[30],'last_retract':summary[-1]},ensure_ascii=False))
    finally:e.close()


if __name__=='__main__':main()
