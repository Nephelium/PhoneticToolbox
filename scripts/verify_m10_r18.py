"""Native local-offset sweeps plus full dental/floor display contact."""
from pathlib import Path
import hashlib
import json
import subprocess
from workbench_source import configure
from verification_artifacts import verification_run,write_json
configure()
from phonetic_core.vocal_tract.engine import Engine
ROOT=Path(__file__).resolve().parents[1]
with verification_run('m10-r18','geometry','zero-offset R17 baseline and local-offset contact sweeps') as (out,scratch):
    e=Engine(resource_dir=ROOT/'resources/vocal_tract/native');states={};matches={}
    try:
        baseline=json.loads((ROOT/'tests/fixtures/m10-r17-zero-shape-digests.json').read_text('utf-8'))
        for name in ['a','i','u','s','l','t']:
            s=e.snapshot(e.presets[name]);matches[name]={k:hashlib.sha256(json.dumps(s[k],sort_keys=True,separators=(',',':')).encode()).hexdigest()==v for k,v in baseline[name].items()}
            assert all(matches[name].values()),name
            for shape in [[0]*4,[-.5,-.3,-.5,-.4],[.5,.8,.5,.4],[0,.8,0,-.4],[0,-.3,0,.4]]:
                states[f'{name}-{shape}']=e.snapshot(e.presets[name],tongue_shape=shape)
        for jaw in [-7,-3.95,0]:
            for shape in [[-.5,-.3,-.5,-.4],[.5,.8,.5,.4]]:
                p=list(e.presets['a']);p[3]=jaw;p[10]=7;p[11]=0
                states[f'jaw{jaw}-{shape}']=e.snapshot(p,tongue_shape=shape)
        write_json(out/'baseline.json',{'matches':matches,'native_sha256':hashlib.sha256((ROOT/'resources/vocal_tract/native/geometry_p2.dll').read_bytes()).hexdigest()})
        write_json(scratch/'states.json',states)
        subprocess.run(['node',str(ROOT/'scripts/verify_m10_jaw_geometry.mjs'),str(scratch/'states.json'),str(out/'report.json')],cwd=ROOT,check=True)
        print(out/'report.json')
    finally:e.close()
