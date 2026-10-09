"""Independent blade/tip combinations around the user-reported poses."""
import itertools
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packages/phonetic_core/src'))
from phonetic_core.vocal_tract.engine import Engine

out=ROOT/'output/validation/m10-r14/combinations';out.mkdir(exist_ok=True)
e=Engine(resource_dir=ROOT/'resources/vocal_tract/native')
try:
    cases=[{'name':n,'state':e.snapshot(p)} for n,p in e.presets.items()]
    for tx,ty,bx,by in itertools.product([1.5,2.1,3.2],[.25,.6],[1.6,2.6,3.6],[-2.5,-1.3,-.3]):
        p=e.presets['a'].copy();p[9]=-2.1;p[10]=tx;p[11]=ty;p[12]=bx;p[13]=by
        cases.append({'name':f'coupled-{tx}-{ty}-{bx}-{by}','state':e.snapshot(p)})
    (out/'current.json').write_text(json.dumps(cases),'utf-8')
    (out/'baseline.json').write_bytes((out.parent/'r12-baseline.json').read_bytes())
    print(json.dumps({'poses':len(cases),'out':str(out)}))
finally:e.close()
