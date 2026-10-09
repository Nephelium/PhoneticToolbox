"""Native closure fixtures and dense, non-audible M10-R11 shape evidence."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packages/phonetic_core/src'))
from phonetic_core.vocal_tract.engine import Engine

out=ROOT/'output/validation/m10-r11';out.mkdir(exist_ok=True,parents=True)
e=Engine(resource_dir=ROOT/'resources/vocal_tract/native')
records=[];examples=[]
try:
    for name in ['a','i','u','n','t','l']:
        for vo in [0,.5,1,1.5]:
            for raise_by in [0,.2,.4,.6]:
                p=e.presets[name].copy();p[7]=vo;p[9]+=raise_by
                s=e.snapshot(p)
                sealed=[i for i,x in enumerate(s['airway_sections']) if i>16 and x['area']<1e-8]
                lateral=[i for i,x in enumerate(s['airway_sections']) if i>16 and x['area']>.01 and (x['upper'][48] is None or x['lower'][48] is None or x['upper'][48]-x['lower'][48]<=1e-6)]
                if sealed or lateral:
                    records.append({'preset':name,'VO':vo,'TCY_delta':raise_by,'sealed':sealed,'lateral':lateral})
                if sealed and (name in ['a','u'] and raise_by>0 or name=='t' and vo==0 and raise_by==0):
                    examples.append({'name':f'{name}-{vo}-{raise_by}','state':s})
    (out/'closure-search.json').write_text(json.dumps(records,indent=2),'utf-8')
    (out/'closure-examples.json').write_text(json.dumps(examples,separators=(',',':')),'utf-8')
    print(json.dumps(records),flush=True)
    # Keep fields consumed by geometry verification; avoid audio or device I/O.
    cases=[]
    for name,steps in [('a',151),('i',31),('u',31),('n',31)]:
        for vo in np.linspace(0,1.5,steps):
            p=e.presets[name].copy();p[7]=vo;s=e.snapshot(p)
            cases.append({'name':f'{name}-{vo:.3f}','state':{k:s[k] for k in ['meshes','contours','airway_sections','centerline','nasal']}})
    (out/'sweep.json').write_text(json.dumps(cases,separators=(',',':')),'utf-8')
    print(f'{len(cases)} continuous native poses saved')
finally:e.close()
