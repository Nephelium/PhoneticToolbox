"""M01-B supplemental conversion evidence, original interpreter and original function.

Stops before REAPER execution: the Python callback only reads the converted WAV.
This proves bytes, not native execution. Existing baselines are never overwritten.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.io import wavfile
from baseline_support import sha

ROOT = Path(__file__).resolve().parents[1]


def samples(rate, dtype, channels):
    t = np.arange(1600) / rate
    signal = .4*np.sin(2*np.pi*120*t)
    if channels == 2: signal = np.column_stack([signal, .2*np.cos(2*np.pi*180*t)])
    if dtype == 'uint8': return (signal*128+128).astype(dtype)
    if dtype == 'int16': return (signal*32768).astype(dtype)
    if dtype == 'int32': return (signal*2147483648).astype(dtype)
    return signal.astype(dtype)


WORKER = r'''
import hashlib,json,sys
from pathlib import Path
from scipy.io import wavfile
assert sys.dont_write_bytecode
request=json.loads(Path(sys.argv[1]).read_text('utf-8'))
sys.path.insert(0,request['source'])
from phonetic_toolbox.core.acoustic import f0_reaper, reaper_python
assert Path(f0_reaper.__file__).resolve().is_relative_to(Path(request['source']).resolve())
def unavailable(*a,**kw): raise RuntimeError('conversion-only probe')
f0_reaper._find_reaper_bin=unavailable
rows=[]
for case in request['cases']:
 def capture(path,*a,**kw):
  fs,data=wavfile.read(path)
  case['pcm_sha256']=hashlib.sha256(data.tobytes()).hexdigest()
  case['wav_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
  case['output_frames']=len(data);case['output_rate']=fs;case['output_dtype']=str(data.dtype)
  return [],[],[]
 reaper_python.run_python_impl=capture
 f0_reaper.compute_reaper_f0(Path(case['filename']),.005,60.,880.,True,False)
 assert 'pcm_sha256' in case
 rows.append(case)
Path('captured.json').write_text(json.dumps(rows),encoding='utf-8')
'''


def main():
    target=ROOT/'tests/fixtures/m01-conversion.json'
    if target.exists(): raise FileExistsError('Frozen conversion evidence already exists')
    folder=ROOT/'output/validation/m01'/('conversion-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    folder.mkdir(parents=True,exist_ok=False)
    evidence=json.loads((ROOT/'docs/baseline/local-evidence.json').read_text('utf-8'))
    source=Path(evidence['baseline']['root'])
    context=json.loads((ROOT/'output/validation/m01/context-before-b.json').read_text('utf-8'))
    interpreter=Path(context['v2_environment'])/'python.exe'
    source_file=source/'phonetic_toolbox/core/acoustic/f0_reaper.py'
    source_hash=sha(source_file)
    cases=[]
    for rate in [16000,22050,44100]:
        for dtype in ['uint8','int16','int32','float32','float64']:
            for channels in [1,2]:
                name=f'{rate}-{dtype}-{channels}.wav'
                wavfile.write(folder/name,rate,samples(rate,dtype,channels))
                cases.append(dict(rate=rate,dtype=dtype,channels=channels,filename=name,input_sha256=sha(folder/name)))
    request=folder/'request.json'
    request.write_text(json.dumps({'source':str(source),'cases':cases}),encoding='utf-8')
    env=dict(os.environ,TEMP=str(folder),TMP=str(folder))
    env['PATH']=str(interpreter.parent/'Library/bin')+os.pathsep+env['PATH']
    completed=subprocess.run([str(interpreter),'-B','-X','utf8','-c',WORKER,str(request)],
        cwd=folder,env=env,text=True,encoding='utf-8',capture_output=True,timeout=90)
    (folder/'worker.log').write_text(completed.stdout+completed.stderr,encoding='utf-8')
    completed.check_returncode()
    rows=json.loads((folder/'captured.json').read_text('utf-8'))
    assert len(rows)==30 and sha(source_file)==source_hash
    assert all(sha(folder/c['filename'])==c['input_sha256'] for c in rows)
    output={'producer':'original-v2-conversion-only','source_sha256':source_hash,
        'recorder_sha256':sha(__file__),'cases':rows,'native_execution':False}
    target.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print('30 independent conversion byte records frozen; no native execution')


if __name__=='__main__': main()
