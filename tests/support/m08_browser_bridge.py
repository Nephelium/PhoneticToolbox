"""Test-only stdio fixture: real M08 compute, synthetic owned files, no host/DB.

Used by the existing Vite development/test runner pattern. This is deliberately
not a production task adapter and does not prove owner/quota/executor wiring.
"""
import base64
import hashlib
import json
import sys
from pathlib import Path
from uuid import uuid4
import numpy as np
import parselmouth
from ptb_worker.m08_jobs import execute
from phonetic_core.manipulation.m08_rules import track, next_name

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/validation/m08'/('browser-'+uuid4().hex[:8])
OUT.mkdir(parents=True); SAVED=OUT/'saved';SAVED.mkdir()
files={};results={};jobs=[];saved={};keys={}
for i,name in enumerate(['public ɑ̃˥.wav','second.wav']):
    t=np.arange(16000)/16000
    snd=parselmouth.Sound(.3*np.sin(2*np.pi*(150+i*30)*t),16000)
    path=OUT/name;snd.save(str(path),'WAV');files[str(i)]=path
broken=OUT/'broken.wav';broken.write_bytes(b'not a WAV');files['2']=broken


def public_file(id,path):return dict(id=id,name=path.name,size=path.stat().st_size,kind='audio',sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def binary(path):return base64.b64encode(path.read_bytes()).decode()


def dispatch(data):
    op=data['op']
    if op=='list':return [public_file(i,p) for i,p in files.items()]
    if op=='preview':
        path=files[data['id']];info=execute(parselmouth.Sound(str(path)),dict(action='preview'),path.stem)
        return dict(info,wav=binary(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if op=='jobs':return jobs
    if op=='submit':
        if data['key'] in keys:return keys[data['key']]
        path=files[data['id']];job=dict(id=str(uuid4()),source_id=data['id'],state='running',results=[])
        jobs.append(job);keys[data['key']]=job
        def emit(name,sound,meta):
            rid=str(uuid4());destination=OUT/(rid+'.wav');sound.save(str(destination),'WAV')
            times,f0=track(parselmouth.Sound(str(destination)))
            result=dict(id=rid,name=name,source_id=data['id'],start=meta['source_start_s'],end=meta['source_end_s'],config=data['config'],times=times.tolist(),original_f0=f0.tolist())
            results[rid]=(destination,result);job['results'].append(result)
            if data['config']['action'] in ('linear','transform'):save_result(rid)
            return rid
        try:
            snd=parselmouth.Sound(str(path))
            execute(snd,data['config'],path.stem,emit=emit);job['state']='succeeded'
        except Exception as e:job['state']='failed';job['error']='m08_audio_decode_failed' if isinstance(e,parselmouth.PraatError) else str(e)
        return job
    if op=='audio':return binary(results[data['id']][0])
    if op=='save':return save_result(data['id'])
    if op=='history':return [r for _,r in saved.values() if r['source_id']==data['id']]
    if op=='remove':
        removed=[]
        for id in data['ids']:
            path,result=saved[id]
            assert path.parent==SAVED and path.is_relative_to(OUT)
            path.unlink();saved.pop(id);removed.append(id)
        return dict(removed=removed,failed=[])
    if op=='rename':
        proposed=[]
        for change in data['changes']:
            old,result=saved[change['id']];name=change['name']
            if Path(name).name!=name or any(c in name for c in '/\\:') or not name.endswith('.wav'):raise ValueError('invalid_name')
            new=SAVED/name
            if new.exists() and new!=old:raise ValueError('name_conflict')
            proposed.append((change['id'],old,new,result))
        if len({n.name.lower() for _,_,n,_ in proposed})!=len(proposed):raise ValueError('duplicate_name')
        for id,old,new,result in proposed:
            if old!=new:old.rename(new)
            result=dict(result,name=new.name);saved[id]=(new,result)
        return [saved[id][1] for id,_,_,_ in proposed]
    if op=='cancel':
        for job in jobs:
            if job['id']==data['id'] and job['state'] in ('queued','running'):job['state']='cancelled'
        return None
    raise ValueError('unknown_operation')


def save_result(id):
    path,r=results[id];stem=files[r['source_id']].stem
    # Real file creation uses exclusive mode; no overwrite even in this test adapter.
    name=next_name(stem,r['start'],r['end'],[p.name for p in SAVED.iterdir()]) if r['config']['action']=='synthesize' else r['name']
    if (SAVED/name).exists():name=next_name(stem,r['start'],r['end'],[p.name for p in SAVED.iterdir()])
    destination=SAVED/name
    with destination.open('xb') as f:f.write(path.read_bytes())
    rid=str(uuid4());result=dict(r,id=rid,name=name);saved[rid]=(destination,result);results[rid]=(destination,result)
    return result


print(json.dumps(dict(ready=True,out=str(OUT))),flush=True)
for line in sys.stdin:
    data=json.loads(line)
    try:payload=dict(id=data['rpc_id'],value=dispatch(data))
    except Exception as e:payload=dict(id=data['rpc_id'],error=str(e))
    print(json.dumps(payload,ensure_ascii=True),flush=True)
