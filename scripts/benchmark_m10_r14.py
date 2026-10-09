"""Real native 129/257 section comparison, no audio/device access."""
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packages/phonetic_core/src'))
import phonetic_core.vocal_tract.engine as module

resource=ROOT/'resources/vocal_tract/native'
libraries=module.resolve_libraries(resource)
out=ROOT/'output/validation/m10-r14'
results=[];snapshots=[]
for count,dll in [(129,out/'resolution129/geometry_p2.dll'),(257,libraries[2])]:
    module.resolve_libraries=lambda _:(libraries[0],libraries[1],dll)
    e=module.Engine(resource_dir=resource)
    try:
        poses=[e.presets[n].copy() for n in ['a','i','u','n','l','s']]
        p=e.presets['a'].copy();p[9]=-2.1;p[10]=1.5;p[11]=.25;p[12]=2.6;p[13]=-1.3;poses.append(p)
        for p in poses:e.snapshot(p)
        times=[];states=[]
        for run in range(5):
            for p in poses:
                start=time.perf_counter();s=e.snapshot(p);times.append((time.perf_counter()-start)*1000)
                if run==0:states.append(s)
        assert len(s['centerline'])==count
        snapshots.append(states)
        results.append({'sections':count,'samples':len(times),'median_ms':float(np.median(times)),
                        'p95_ms':float(np.percentile(times,95)),
                        'median_spacing_mm':float(np.median(np.diff(states[0]['centerline'],axis=0)[:,2])*10),
                        'snapshot_bytes':len(json.dumps(states[0]).encode())})
    finally:e.close()
comparisons=[]
for name,a,b in zip(['a','i','u','n','l','s','retroflex'],*snapshots):
    assert a['contours']==b['contours']
    comparisons.append({'pose':name,'max_area_delta_cm2':float(np.max(np.abs(np.array(a['tube_areas'])-b['tube_areas']))),
                        'resonances_129':a['resonances'],'resonances_257':b['resonances']})
report={'runs':results,'comparisons':comparisons,'timing_scope':'native snapshot only; excludes Qt, transport and rendering'}
(out/'resolution-benchmark.json').write_text(json.dumps(report,indent=2),'utf-8')
print(json.dumps(report))
