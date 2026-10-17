"""Finite geometry/audio smoke and portable readout, without sound devices or WAVs."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

parser=argparse.ArgumentParser()
parser.add_argument('--resource-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'packages/phonetic_core/src'))
import numpy as np
from phonetic_core.vocal_tract.engine import Engine

started=time.perf_counter();engine=Engine(resource_dir=args.resource_dir)
report={'platform':platform.platform(),'scope':'native smoke only; no HTTP, load or full parity claim','poses':{}}
try:
    report['geometry_method']=engine.geometry_method
    for name in ('a','i','u'):
        begin=time.perf_counter();p=engine.presets[name]
        state=engine.snapshot(p)
        tube=engine.prepare_tube(p);g=engine.glottis(f0=125)
        engine.reset_tube(tube,g)
        audio=np.concatenate([engine.block_tube(tube,g,480) for _ in range(20)])
        assert np.isfinite(audio).all() and np.std(audio)>1e-6
        report['poses'][name]={'tube_areas':state['tube_areas'],'tube_lengths':state['tube_lengths'],'nasal_area':state['nasal_area'],'resonances':state['resonances'],'samples':len(audio),'rms':float(np.sqrt(np.mean(audio**2))),'peak':float(np.max(np.abs(audio))),'elapsed_seconds':time.perf_counter()-begin}
    report['sample_rate']=engine.sr
finally:
    engine.close()
report['elapsed_seconds']=time.perf_counter()-started
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'ok':True,'platform':report['platform'],'elapsed_seconds':report['elapsed_seconds']}))
