"""Jaw-bound ventral display, preserving native acoustic geometry."""
from pathlib import Path
import subprocess
from workbench_source import configure
from verification_artifacts import verification_run,write_json
configure()
from phonetic_core.vocal_tract.engine import Engine

ROOT=Path(__file__).resolve().parents[1]
with verification_run('m10-r17','jaw','129 sections; six presets, jaw/tip grid, curled blade and side edges; full triangle-floor intersections and dorsal preservation') as (out,scratch):
    e=Engine(resource_dir=ROOT/'resources/vocal_tract/native');states={}
    try:
        for name in ['a','i','u','s','l','t']:
            states[name]=e.snapshot(e.presets[name])
        for jaw in [-7,-3.95,0]:
            for x in [2,3.5,5,7]:
                for y in [-2,0,2]:
                    p=list(e.presets['a']);p[3]=jaw;p[10]=x;p[11]=y;p[12]=x-.4;p[13]=-3
                    states[f'jaw{jaw}-tip{x}-{y}']=e.snapshot(p)
        for side in [-1,0,1]:
            p=list(e.presets['l']);p[18]=side
            states[f'side{side}']=e.snapshot(p)
        write_json(scratch/'states.json',states)
        subprocess.run(['node',str(ROOT/'scripts/verify_m10_jaw_geometry.mjs'),str(scratch/'states.json'),str(out/'report.json')],cwd=ROOT,check=True)
        print(out/'report.json')
    finally:e.close()
