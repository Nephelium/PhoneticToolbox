"""R17 native gaps, bounded blade, fixed floor and exact section optimization."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from workbench_source import configure
configure()
from verification_artifacts import verification_run,write_json
import phonetic_core.vocal_tract.engine as module

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path);args=parser.parse_args()
    if args.candidate:
        original=module.resolve_libraries
        module.resolve_libraries=lambda p:(*original(p)[:2],args.candidate.resolve())
    with verification_run('m10-r17','geometry','JD2 65/129/257; high tongue, depressed blade, side controls; exact reduced nasal section versus full mesh') as (out,scratch):
        e=module.Engine(resource_dir=ROOT/'resources/vocal_tract/native');report={'passed':False,'presets':[],'recovered':[]};states={}
        try:
            for count in [65,129,257]:
                e.set_section_count(count)
                for name in ['a','i','u','s','l','t']:
                    s=e.snapshot(e.presets[name]);assert s['geometry_method']=='m10/r17'
                    assert np.isfinite(s['tube_areas']).all()
                    assert s['tongue_floor_rib']==10
                    for sec,area in zip(s['airway_sections'],s['raw_areas']):
                        assert area>=sec['area']-2e-5,(name,count,sec['index'],area,sec['area'])
                    report['presets'].append({'name':name,'sections':count,'ms':s['compute_ms'],'min_area':s['oral_min_area']})
                    if count==129:states[name]=s
                p=list(e.presets['i']);p[9]=-.2;p[18]=0;s=e.snapshot(p)
                for i,sec in enumerate(s['airway_sections']):
                    up=np.empty(96);lo=np.empty(96);e.geom.p0_profile(i,module.dp(up),module.dp(lo))
                    valid=(abs(up)<100)&(abs(lo)<100)&(up>=lo)
                    old=float(np.trapezoid(np.where(valid,up-lo,0))*7/96)
                    if old<1e-6 and sec['area']>.005:
                        assert s['raw_areas'][i]>.005
                        report['recovered'].append({'sections':count,'index':i,'old_profile_area':old,'geometric_area':sec['area'],'acoustic_area':s['raw_areas'][i]})
                if count==129:states['high']=s
            assert len(report['recovered'])>=3
            e.set_section_count(129);anchor=states['a']['contours']['lower_cover'][10]
            for k,(x,y) in enumerate([(2,2.5),(3.5,1.5),(5,1),(7,0)]):
                p=list(e.presets['a']);p[10]=x;p[11]=y;p[12]=x-.4;p[13]=-3
                s=e.snapshot(p);states['anchor-'+str(k)]=s
                assert s['contours']['lower_cover'][10]==anchor
                assert s['limited'][13]>p[13]+.5
                assert np.isfinite(next(m['vertices'] for m in s['meshes'] if m['name']=='tongue')).all()
            p=list(e.presets['a']);p[3]=0;s=e.snapshot(p)
            assert s['contours']['lower_cover'][10]!=anchor;states['jaw']=s
            path=scratch/'states.json';write_json(path,states)
            subprocess.run(['node',str(ROOT/'scripts/verify_m10_r16_geometry.mjs'),str(path),str(out/'contact.json')],cwd=ROOT,check=True)
            subprocess.run(['node',str(ROOT/'scripts/verify_m10_r17_geometry.mjs'),str(path),str(out/'render.json')],cwd=ROOT,check=True)
            report['passed']=True
        finally:e.close();write_json(out/'report.json',report)
        print(out/'report.json')

if __name__=='__main__':main()
