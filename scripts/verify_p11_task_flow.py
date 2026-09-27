"""Actual loopback HTTP + existing-schema SQLite copy + owned Linux task groups.

No DDL, no service database, no external ports. Inputs are public synthetic data.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import threading
import time
from uuid import uuid4


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['lpc','egg','acoustic'],required=True)
    parser.add_argument('--faults',action='store_true')
    parser.add_argument('--reaper',type=Path)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import pandas as pd
    from scipy.io import wavfile
    import httpx
    import uvicorn
    from ptb_api.main import create_app
    from ptb_worker.store import SQLiteJobStore,LOCAL_PROJECT
    from ptb_worker.local_acoustic_files import LocalAcousticFiles,initialize_local_files
    from ptb_worker.acoustic_batches import AcousticBatches
    from ptb_worker.acoustic_executor import execute_acoustic_claim
    if os.name=='nt':
        assert args.reaper and args.reaper.is_file()
        profile=dict(reaper_binary=str(args.reaper.resolve()))
    else:
        from ptb_worker.native.linux_runtime import load_profile
        _,profile=load_profile()
    with sqlite3.connect(args.template.resolve().as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(out/'jobs.sqlite3') as dst:
        assert not src.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
        src.backup(dst)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    store=SQLiteJobStore(out/'jobs.sqlite3',max_running=1)
    files=LocalAcousticFiles(store,cache,reaper_binary=profile.get('reaper_binary'))
    AcousticBatches(store,files)
    token=uuid4().hex;origin='http://127.0.0.1'
    app=create_app('local',job_store=store,local_token=token,local_origin=origin)
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'))
    thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);thread.start()
    deadline=time.monotonic()+10
    while not server.started:
        assert thread.is_alive() and time.monotonic()<deadline
        time.sleep(.02)
    profile_sha=hashlib.sha256(Path(os.environ['PTB_LINUX_RUNTIME_PROFILE']).read_bytes()).hexdigest() if os.name!='nt' else None
    report=dict(phase=args.phase,success=False,schema_applied=[],jobs=[],faults=[],transport='loopback HTTP / local-token / durable SQLite copy',profile_sha256=profile_sha)
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{port}',headers={'Authorization':'Bearer '+token,'Origin':origin},timeout=30,trust_env=False) as client:
            assert httpx.get(f'http://127.0.0.1:{port}/api/v1/jobs',params={'project_id':LOCAL_PROJECT},trust_env=False).status_code==403
            root=Path(__file__).resolve().parents[1]
            with np.load(root/'tests/fixtures/m03/EGG-SYN-PCM16.npz') as data:
                samples=np.column_stack([data['load.egg_signal_raw'],data['load.audio_signal']])
            if args.phase=='acoustic':samples=samples[:,1]
            stream=io.BytesIO();wavfile.write(stream,44100,samples);source=stream.getvalue()
            if os.name!='nt':
                response=client.post('/api/v1/preview/spectrogram',params=dict(channel=0,start=0.,end=.5,width=200),content=source,headers={'Content-Type':'application/octet-stream'})
                assert response.status_code==200,response.text
                assert response.json()['sha256']==hashlib.sha256(source).hexdigest()
                report['spectrogram_preview']=True
            response=client.post('/api/v1/jobs/local-inputs',params={'role':'audio','name':'合成 ɑ̃˥.wav'},content=source)
            assert response.status_code==200,response.text
            ref=response.json()
            font=dict(zh='Noto Sans SC',latin='DejaVu Sans')
            if args.phase!='acoustic':
                response=client.post('/api/v1/jobs/'+args.phase+'/fonts',json=font)
                assert response.status_code==200 and response.json()['available'],response.text
                report['font_preflight']=response.json()

            def submit(config=None):
                if args.phase=='acoustic':
                    body=dict(operation='acoustic_analysis',project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,inputs=[{'audio':ref}],config=config or {})
                    response=client.post('/api/v1/jobs/batches/create',json=body)
                    assert response.status_code==201,response.text
                    claim=store.claim('p11-task-check');assert claim
                    return store.get('local',claim['id']),claim
                body=dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,audio=ref,config=config or ({'roi_end':.05,'font':font} if args.phase=='lpc' else {'mode':'batch'}))
                route='/api/v1/jobs/'+args.phase+'/create'
                response=client.post(route,json=body);assert response.status_code==201,response.text
                assert client.post(route,json=body).json()['id']==response.json()['id']
                job=response.json();claim=store.claim('p11-task-check');assert claim['id']==job['id']
                return job,claim

            def execute(job,claim,hook=None):
                evidence={};started=time.monotonic()
                execute_acoustic_claim(store,claim,'p11-task-check',threading.Event(),on_started=hook,process_evidence=evidence)
                result=client.get('/api/v1/jobs/'+job['id']).json()
                record=dict(id=job['id'],state=result['state'],error=result['error_code'],elapsed_seconds=time.monotonic()-started,process=evidence)
                if os.name!='nt':record['profile_sha256']=hashlib.sha256(Path(os.environ['PTB_LINUX_RUNTIME_PROFILE']).read_bytes()).hexdigest()
                if os.name!='nt':assert evidence.get('cleaned') is True,record
                if evidence.get('main_pid'):assert not Path('/proc',str(evidence['main_pid'])).exists(),record
                report['jobs'].append(record)
                return result

            def readback(done):
                assert done['state']=='succeeded',done
                target=out/done['id'];target.mkdir();returned={}
                for entry in done['result_manifest']['files']:
                    value=bytearray()
                    for offset in range(0,entry['size_bytes'],1048576):
                        r=client.get('/api/v1/jobs/local-results/'+entry['id'],params=dict(offset=offset,size=min(1048576,entry['size_bytes']-offset)))
                        assert r.status_code==200,r.text
                        value.extend(r.content)
                    raw=bytes(value);assert hashlib.sha256(raw).hexdigest()==entry['sha256']
                    returned[entry['name']]=raw;(target/entry['name']).write_bytes(raw)
                    if entry['name'].endswith('.png'):
                        from PIL import Image
                        with Image.open(io.BytesIO(raw)) as image:image.verify()
                return returned

            configs=([{'roi_end':.05,'font':font},{'roi_start':.4,'roi_end':.5,'font':font,'dynamic_y':True}] if args.phase=='lpc' else
                     [{'mode':'single','roi_end':.5,'font':font},{'mode':'batch','generate_images':True,'font':font},{'mode':'inverse','roi_end':.12}] if args.phase=='egg' else
                     [{},{'backend_policy':{'reaper':'native_required'}}])
            for config in configs:
                job,claim=submit(config);done=execute(job,claim);blobs=readback(done)
                if args.phase=='lpc':
                    meta=json.loads(blobs['lpc.ptb.json']);rate,values=wavfile.read(io.BytesIO(blobs['lpc_AUDIO.wav']))
                    first,last=meta['selection']['start_sample'],meta['selection']['end_sample']
                    assert rate==44100 and np.array_equal(values,np.mean(samples[first:last].astype(float),axis=1))
                elif args.phase=='egg':
                    meta=json.loads(blobs['egg.ptb.json'])
                    if config['mode']=='inverse':
                        rate,values=wavfile.read(io.BytesIO(blobs['egg_IF.wav']));assert rate==44100 and len(values)==5292 and np.isfinite(values).all()
                    else:
                        frame=pd.read_csv(io.BytesIO(blobs['egg_DATA.csv']));assert len(frame)>0 and frame['CQ'].notna().any()
                else:
                    from ptb_api.acoustic_models import AcousticResult
                    model=AcousticResult.model_validate_json(blobs['result.ptb.json'])
                    meta=model.model_dump()
                    assert any(b.actual=='native_reaper' for b in model.metadata.backends),meta['metadata']['backends']
                    assert next(c for c in model.numeric if c.key=='rF0').nonfinite.count(0)>0
                    frame=pd.read_excel(io.BytesIO(blobs['result.xlsx']))
                    report['acoustic_columns']=list(frame.columns)
                    assert any('rF0' in str(c) and frame[c].notna().any() for c in frame.columns)
                    db=target_db=out/(job['id']+'.sqlite')
                    db.write_bytes(blobs['result.ptb.sqlite'])
                    with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as conn:assert conn.execute('PRAGMA integrity_check').fetchone()==('ok',)
            if args.faults:
                # A partial output must remain invisible and be reclaimed.
                from ptb_api.quota import StorageError
                job,claim=submit();original_write=files.write;writes=[0]
                def fail_write(identity,asset_id,offset,raw):
                    with files.locked() as state:kind=state['assets'][asset_id]['kind']
                    if kind=='result':
                        writes[0]+=1
                        if writes[0]==2:raise StorageError('storage_write_failed',503)
                    return original_write(identity,asset_id,offset,raw)
                files.write=fail_write
                try:done=execute(job,claim)
                finally:files.write=original_write
                assert writes[0]==2 and done['state']=='failed' and not done['result_manifest'],done
                report['faults'].append('partial result write rollback')
                job,claim=submit()
                done=execute(job,claim,lambda pid:store.cancel('local',job['id']))
                assert done['state']=='cancelled' and not done['result_manifest'],done
                report['faults'].append('live cancellation')
                job,claim=submit()
                done=execute(job,claim,lambda pid:os.kill(pid,signal.SIGKILL))
                assert done['state']=='failed' and not done['result_manifest'],done
                report['faults'].append('actual child SIGKILL')
                if args.phase=='acoustic':
                    saved=files.reaper_binary;files.reaper_binary=str(out/'missing-reaper')
                    job,claim=submit({'backend_policy':{'reaper':'native_required'}})
                    try:done=execute(job,claim)
                    finally:files.reaper_binary=saved
                    assert done['state']=='failed' and not done['result_manifest'],done
                    report['faults'].append('missing native REAPER rejects complete extraction')
                    # Registered but crashing native fixture exercises the legacy
                    # core's caught-error path. It must not publish missing rF0.
                    fake=out/'failing-reaper';fake.write_text('#!/bin/sh\nexit 7\n');fake.chmod(0o700)
                    prior=os.environ['PTB_LINUX_RUNTIME_PROFILE']
                    fault_profile=json.loads(Path(prior).read_text('utf-8'))
                    fault_profile['reaper_binary']=str(fake)
                    fault_profile['hashes'][str(fake)]=hashlib.sha256(fake.read_bytes()).hexdigest()
                    fault_path=out/'fault-runtime.json';fault_path.write_text(json.dumps(fault_profile),encoding='utf-8')
                    os.environ['PTB_LINUX_RUNTIME_PROFILE']=str(fault_path);files.reaper_binary=str(fake)
                    try:
                        job,claim=submit();done=execute(job,claim)
                    finally:
                        os.environ['PTB_LINUX_RUNTIME_PROFILE']=prior;files.reaper_binary=saved
                    assert done['state']=='failed' and done['error_code']=='reaper_runtime_failed' and not done['result_manifest'],done
                    report['faults'].append('registered native exit failure cannot publish missing rF0')
                if args.phase=='lpc':
                    from ptb_worker.native import linux_runtime
                    original=linux_runtime.command
                    job,claim=submit()
                    linux_runtime.command=lambda entry,request:[profile['python'],'-c','import time; time.sleep(60)']
                    try:done=execute(job,claim)
                    finally:linux_runtime.command=original
                    assert done['state']=='failed' and done['error_code']=='deadline_exceeded',done
                    report['faults'].append('real 30 second timeout with controlled stalled child')
                job,claim=submit();readback(execute(job,claim));report['faults'].append('next actual task recovers')
            state=json.loads((cache/'.ptb-local.json').read_text('utf-8'))
            assert all(a['state']=='deleted' for a in state['assets'].values() if a['kind']=='temporary')
            report.update(success=True,temporary_files_remaining=0)
    finally:
        server.should_exit=True;thread.join(10);sock.close()
        report['http_stopped']=not thread.is_alive()
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
