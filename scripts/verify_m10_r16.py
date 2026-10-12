"""Native and display R16 geometry, with lifecycle-managed snapshot inputs."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from workbench_source import configure
configure()
from verification_artifacts import verification_run
import phonetic_core.vocal_tract.engine as module

ROOT=Path(__file__).resolve().parents[1]


def runs(sec):
    groups=[]
    for i,(u,l) in enumerate(zip(sec['upper'],sec['lower'])):
        if u is None or l is None or u-l<=1e-5:continue
        if not groups or groups[-1][-1]!=i-1:groups.append([])
        groups[-1].append(i)
    return groups


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path);args=parser.parse_args()
    if args.candidate:
        original=module.resolve_libraries
        module.resolve_libraries=lambda p:(*original(p)[:2],args.candidate.resolve())
    with verification_run('m10-r16','geometry','JD2, native 65/129/257; s/l/t topology, tongue/jaw sweep, JS air mesh/tooth intersections; no physical audio') as (out,scratch):
        e=module.Engine(resource_dir=ROOT/'resources/vocal_tract/native');report={'passed':False,'native':[]};states={}
        try:
            for count in [65,129,257]:
                e.set_section_count(count)
                for name in ['a','i','u','s','l','t']:
                    s=e.snapshot(e.presets[name]);assert np.isfinite(s['tube_areas']).all()
                    profiles=[(sec,runs(sec)) for sec in s['airway_sections']]
                    front=[(sec,rs) for sec,rs in profiles if sec['index']>count*.7]
                    if name=='l':assert any(len(rs)==2 and rs[0][-1]<48<rs[1][0] for sec,rs in front)
                    if name=='s':assert any(len(rs)==1 and rs[0][0]<=48<=rs[0][-1] and rs[0][-1]-rs[0][0]<15 and 0<sec['area']<.15 for sec,rs in front)
                    if name=='t':assert any(not rs for sec,rs in front)
                    report['native'].append({'pose':name,'sections':count,'min_area':s['oral_min_area'],'split_sections':[sec['index'] for sec,rs in front if len(rs)==2]})
                    if count==129:states[name]=s
            e.set_section_count(129)
            # Dense side sweeps exercise station remapping and both sides of
            # the old saturation boundary without a prescribed phone label.
            report['side_sweeps']=[]
            for name in ['TS1','TS2','TS3']:
                previous=None;jumps=[];anchors=[];side_motion=[0.,0.]
                index=e.names.index(name)
                for value in np.linspace(-1,1,81):
                    p=list(e.presets['s']);p[index]=float(value);s=e.snapshot(p)
                    v=np.array(next(m['vertices'] for m in s['meshes'] if m['name']=='tongue'))
                    assert np.isfinite(v).all() and np.isfinite(s['tube_areas']).all()
                    # Bilateral linkage controls both edges. The inherited
                    # 96-point hull grid is offset from zero by half a sample,
                    # so vertex-for-vertex mirror equality is not its contract.
                    mesh=next(m for m in s['meshes'] if m['name']=='tongue')
                    grid=v.reshape(-1,mesh['points'],3)
                    if previous is not None:
                        delta=abs(grid-previous.reshape(grid.shape));jumps.append(float(delta.max()))
                        side_motion[0]+=float(delta[:,:5].max());side_motion[1]+=float(delta[:,-5:].max())
                    previous=v;anchors.append(s['contours']['lower_cover'][s['tongue_floor_rib']])
                assert all(p==anchors[0] for p in anchors)
                assert min(side_motion)>0
                report['side_sweeps'].append(dict(parameter=name,steps=81,max_vertex_step_cm=max(jumps),side_motion_cm=side_motion))
            report['contact_continuity']=[]
            for lo,hi in [(-1e-5,1e-5),(.11375,.114),(.29999,.30001)]:
                previous=None;jumps=[]
                for value in np.linspace(lo,hi,51):
                    p=list(e.presets['s']);p[18]=float(value);s=e.snapshot(p)
                    v=np.array(next(m['vertices'] for m in s['meshes'] if m['name']=='tongue'))
                    if previous is not None:jumps.append(float(abs(v-previous).max()))
                    previous=v
                # Covers the reproduced invalid-contact/grid-width step, not
                # an assertion that all articulatory derivatives are bounded.
                assert max(jumps)<.005
                report['contact_continuity'].append(dict(range=[lo,hi],steps=51,max_vertex_step_cm=max(jumps)))
            for k,(x,y) in enumerate([(3.5,-1),(5,1),(2.5,1.5),(2,2.5),(7,0)]):
                p=list(e.presets['a']);p[10]=x;p[11]=y
                s=e.snapshot(p);states[f'anchor-{k}']=s
                assert np.isfinite(next(m['vertices'] for m in s['meshes'] if m['name']=='tongue')).all()
            p=list(e.presets['a']);p[3]=0;states['jaw-closed']=e.snapshot(p)
            assert states['jaw-closed']['contours']['lower_cover'][12]!=states['anchor-0']['contours']['lower_cover'][12]
            for val in [-1,-.6,-.15,0,.15,.3,.6,1]:
                p=list(e.presets['s']);p[18]=val;states[f'side-{val}']=e.snapshot(p)
            body=scratch/'states.json';body.write_text(json.dumps(states,ensure_ascii=False),encoding='utf-8')
            subprocess.run(['node',str(ROOT/'scripts/verify_m10_r16_geometry.mjs'),str(body),str(out/'display.json')],check=True,cwd=ROOT)
            report['passed']=True
        finally:
            e.close();(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out/'report.json')


if __name__=='__main__':main()
