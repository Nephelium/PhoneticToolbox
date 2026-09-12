"""Probe native poses, then check shared 2D/3D display geometry with Node."""
import json
import subprocess
from pathlib import Path
from phonetic_core.vocal_tract.engine import Engine

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'output/validation/m10/geometry-r4';out.mkdir(parents=True,exist_ok=True)
engine=Engine(resource_dir=ROOT/'resources/vocal_tract/native')
cases=[]
try:
    for name in ('a','i','u'):
        for opening in (0,.2,1.5):
            p=engine.presets[name].copy();p[engine.names.index('LD')]=opening
            cases.append({'name':f'{name}-LD-{opening}','state':engine.snapshot(p)})
    for area in (0,.5,1.5):
        p=engine.presets['n'].copy();p[engine.names.index('VO')]=area
        cases.append({'name':f'n-VO-{area}','state':engine.snapshot(p)})
finally:engine.close()
data=out/'native-poses.json';data.write_text(json.dumps(cases),encoding='utf-8')
subprocess.run(['node',str(ROOT/'scripts/verify_m10_geometry.mjs'),str(data)],cwd=ROOT,check=True)
