"""Describe cross-platform differences without changing the exact acceptance gate."""
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path
import numpy as np
import parselmouth
from phonetic_core.manipulation.m08_synthesis import synthesize_from_pitch
from phonetic_core.manipulation.m08_transform import transform
from phonetic_core.manipulation.m08_batch import generate_batch_linear
from phonetic_core.manipulation.m08_rules import track

root=Path(sys.argv[1]);output=Path(sys.argv[2]);data=np.load(root/'tests/fixtures/m08/v2.npz');meta=json.loads((root/'tests/fixtures/m08/v2.json').read_text())
records=[]
def compare(key,value):
    reference=data[key];actual=np.asarray(value);same_shape=reference.shape==actual.shape
    records.append(dict(key=key,shape=list(actual.shape),exact=bool(same_shape and np.array_equal(reference,actual)),
        max_abs=float(np.max(np.abs(reference-actual))) if same_shape and actual.size else None,
        sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
def seed():parselmouth.praat.run('random_initializeWithSeedUnsafelyButPredictably (42)')
def sound():return parselmouth.Sound(data['samples'],meta['sample_rate'])
times,f0=track(sound());compare('times',times);compare('f0',f0)
with tempfile.TemporaryDirectory(prefix='m08-diagnostic-') as tmp:
    target=Path(tmp)/'actual.wav'
    for i,(start,end,speed) in enumerate(meta['synth']):
        seed();snd=synthesize_from_pitch(sound(),data['times'],data['f0']*1.2,start,end,speed)
        compare(f'synth_{i}',snd.values);compare(f'axis_{i}',[snd.xmin,snd.xmax,snd.dx,snd.x1])
    for i,args in enumerate([(1,1,0),(.8,1,0),(1,1.2,20),(1.5,.8,-10),(1.005,1.005,.005)]):
        if f'transform_{i}' in meta['errors']:continue
        seed();snd=transform(parselmouth.Sound(data['pcm_input'],meta['sample_rate']),*args);snd.save(str(target),'WAV');compare(f'transform_{i}',parselmouth.Sound(str(target)).values)
    for i,case in enumerate(meta['batch']):
        seed();outputs=list(generate_batch_linear(sound(),'public',data['times'],data['f0'],0,.6,**case['args']))
        for j,(_,snd,_) in enumerate(sorted(outputs,key=lambda o:o[0])):
            snd.save(str(target),'WAV');compare(f'batch_{i}_{j}',parselmouth.Sound(str(target)).values)
    seed();snd=next(generate_batch_linear(sound(),'public',data['times'],data['f0'],0,.6,.1,.5,[-20],[20],[],'constant','constant',[],True))[1]
    snd.save(str(target),'WAV');compare('offset',parselmouth.Sound(str(target)).values)
report=dict(platform=platform.platform(),python=sys.version,numpy=np.__version__,parselmouth=parselmouth.__version__,praat=parselmouth.PRAAT_VERSION,records=records,exact=sum(r['exact'] for r in records),total=len(records))
output.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(dict(exact=report['exact'],total=report['total'])));sys.exit(0 if report['exact']==report['total'] else 2)
