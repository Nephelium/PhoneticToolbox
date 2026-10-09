"""Independently sample native/display tissue faces against dorsal tongue triangles."""
import json
import sys
import subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packages/phonetic_core/src'))
from phonetic_core.vocal_tract.engine import Engine


def face_clearance(mesh,tongue,steps=8):
    v=np.asarray(mesh['vertices']).reshape(-1,3)
    faces=v[np.asarray(mesh['triangles']).reshape(-1,3)]
    weights=np.asarray([[i/steps,j/steps,1-(i+j)/steps] for i in range(steps+1) for j in range(steps+1-i)])
    points=np.unique(np.einsum('vk,fkd->fvd',weights,faces).reshape(-1,3),axis=0)
    maximum=np.full(len(points),-np.inf)
    t=np.asarray(tongue['vertices']).reshape(-1,3)
    tri=np.asarray(tongue['triangles']).reshape(-1,3)
    tri=tri[np.all(tri<34*tongue['points'],axis=1)]
    for a,b,c in t[tri]:
        mask=(points[:,0]>=min(a[0],b[0],c[0])-1e-8)&(points[:,0]<=max(a[0],b[0],c[0])+1e-8)&(points[:,2]>=min(a[2],b[2],c[2])-1e-8)&(points[:,2]<=max(a[2],b[2],c[2])+1e-8)
        ids=np.flatnonzero(mask)
        if not len(ids):continue
        d=(b[0]-a[0])*(c[2]-a[2])-(b[2]-a[2])*(c[0]-a[0])
        if abs(d)<1e-12:continue
        q=points[ids]-a
        v=(q[:,0]*(c[2]-a[2])-q[:,2]*(c[0]-a[0]))/d
        w=((b[0]-a[0])*q[:,2]-(b[2]-a[2])*q[:,0])/d
        good=(v>=-1e-8)&(w>=-1e-8)&(v+w<=1+1e-8)
        ids=ids[good];y=a[1]+v[good]*(b[1]-a[1])+w[good]*(c[1]-a[1])
        maximum[ids]=np.maximum(maximum[ids],y)
    valid=np.isfinite(maximum)
    gaps=points[valid,1]-maximum[valid]
    return {'samples':len(points),'over_tongue':int(valid.sum()),'minimum_clearance_cm':float(gaps.min()) if len(gaps) else None,'penetrating':int((gaps < -2e-5).sum())}


def main():
    out=ROOT/'output/validation/m10-r11';e=Engine(resource_dir=ROOT/'resources/vocal_tract/native');cases=[];reports=[]
    try:
        for name in ['a','i','u','e','n','l','t']:
            for vo in [0,.5,.75,1,1.25,1.5]:
                for raised in [0,.6]:
                    p=e.presets[name].copy();p[7]=vo;p[9]+=raised;s=e.snapshot(p)
                    key=f'{name}-vo{vo}-tcy{raised}'
                    u=next(m for m in s['meshes'] if m['name']=='uvula');t=next(m for m in s['meshes'] if m['name']=='tongue')
                    result=face_clearance(u,t)
                    reports.append({'name':key,'lift_cm':s['uvula_contact_lift'],**result})
                    cases.append({'name':key,'state':{k:s[k] for k in ['contours','meshes','nasal','uvula_contact_lift']}})
    finally:e.close()
    (out/'contact-cases.json').write_text(json.dumps(cases,separators=(',',':')),'utf-8')
    (out/'contact-native-report.json').write_text(json.dumps(reports,indent=2),'utf-8')
    failed=[r for r in reports if r['penetrating']]
    print(json.dumps({'cases':len(cases),'failed':failed,'max_lift_cm':max(r['lift_cm'] for r in reports)}))
    assert not failed
    subprocess.run(['node',str(ROOT/'scripts/verify_m10_r11_display_contact.mjs')],check=True,cwd=ROOT)
    display=json.loads((out/'contact-display.json').read_text('utf-8'))
    reports=[{'name':case['name'],**face_clearance(case['mesh'],case['tongue'])} for case in display]
    (out/'contact-display-report.json').write_text(json.dumps(reports,indent=2),'utf-8')
    failed=[r for r in reports if r['penetrating']]
    print(json.dumps({'display_cases':len(reports),'failed':failed}))
    assert not failed


if __name__=='__main__':main()
