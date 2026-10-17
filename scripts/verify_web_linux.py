"""Authorized isolated Linux validation only: fixed dedicated UID/DB/socket/root.

No SMTP, production origin, existing database migration or desktop installation.
Run setup once, then science, then flow. Reports exclude credentials and tokens.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import pwd
import re
import secrets
import subprocess
import sys
import threading
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
STATE=Path('/var/lib/ptb-staging')
DSN='host=/var/lib/ptb-staging/pgsocket port=18767 user=ptb-staging dbname=ptb_web_staging'
OUT=STATE/'reports'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def setup():
    import importlib.metadata as md
    import psycopg
    with psycopg.connect(DSN,autocommit=True) as c:
        assert c.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog','information_schema')").fetchone()[0]==0
        for name in ('001_accounts.sql','002_jobs.sql','003_storage.sql','004_job_assets.sql','005_acoustic_batches.sql'):
            c.execute((ROOT/'backend/migrations'/name).read_text('utf-8'))
        instance=str(uuid4())
        c.execute('INSERT INTO ptb_storage.state(version,instance_id,frozen) VALUES(1,%s,false)',(instance,))
        for name in ('006_storage_policy.sql','008_web_identity.sql','009_web_workspaces.sql','010_web_capacity.sql'):
            c.execute((ROOT/'backend/migrations'/name).read_text('utf-8'))
    (STATE/'assets/.ptb-storage.json').write_text(json.dumps(dict(instance_id=instance)))
    (STATE/'assets/.ptb-storage.lock').write_bytes(b'0')
    private=STATE/'private.json'
    with private.open('x') as f:json.dump(dict(dsn=DSN,storage_root=str(STATE/'assets'),signing_key=secrets.token_urlsafe(48),min_free_bytes=10_000_000_000),f)
    private.chmod(0o600)
    cache=STATE/'matplotlib-cache';cache.mkdir(exist_ok=True)
    paths=[ROOT/'backend/src',ROOT/'packages/phonetic_core/src']
    files=[Path(sys.executable),ROOT/'reaper',ROOT/'fonts/NotoSansSC-VF.ttf']
    for folder in paths:files.extend(p for p in folder.rglob('*.py') if '.egg-info' not in str(p))
    import numpy,scipy
    for module in (numpy,scipy):
        folder=Path(module.__file__).parent
        files.extend(folder.rglob('*.so'));files.extend(folder.parent.glob(module.__name__+'.libs/*.so*'))
    names=['phonetic-core','ptb-api','numpy','scipy','praat-parselmouth','matplotlib','pandas']
    profile=dict(schema='p11-runtime/1',python=sys.executable,cache=str(cache),sys_paths=[str(p) for p in paths],
        versions={n:md.version(n) for n in names},fonts=[str(ROOT/'fonts/NotoSansSC-VF.ttf')],reaper_binary=str(ROOT/'reaper'),
        hashes={str(p):digest(p) for p in files})
    (STATE/'runtime.json').write_text(json.dumps(profile,indent=2))
    assert (STATE/'runtime.json').stat().st_size<=131072
    write('setup.json',dict(versions=profile['versions'],profile_sha256=digest(STATE/'runtime.json'),hash_count=len(files),database='ptb_web_staging',migration='001-006,008-010',public_listener=False))


def environment():
    os.environ.update(PTB_RESOURCE_PROFILE='server-small',PTB_LINUX_RUNTIME_PROFILE=str(STATE/'runtime.json'),
        PTB_LINUX_VALIDATION_RECEIPT=str(STATE/'validated-runtime.json'),MPLCONFIGDIR=str(STATE/'matplotlib-cache'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')


def science():
    receipt=dict(profile_sha256=digest(STATE/'runtime.json'),operations={})
    for phase,operation in [('lpc','lpc_analysis'),('egg','egg_analysis'),('acoustic','acoustic_analysis')]:
        out=OUT/('science-'+phase)
        previous=out/'report.json'
        if previous.is_file():
            value=json.loads(previous.read_text())
            if value.get('success') and value.get('profile_sha256')==receipt['profile_sha256']:
                receipt['operations'][operation]=dict(report=str(previous),sha256=digest(previous))
                continue
            attempt=2
            while out.exists():
                out=OUT/('science-'+phase+'-'+str(attempt));attempt+=1
        with (OUT/('science-'+phase+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/verify_p11_task_flow.py'),'--template',
                str(ROOT/'output/validation/p11-tasks-20260926/synthetic-template.sqlite3'),'--output',str(out),'--phase',phase,'--faults'],stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,('science failed',phase)
        report=out/'report.json';value=json.loads(report.read_text());assert value['success']
        receipt['operations'][operation]=dict(report=str(report),sha256=digest(report))
        print('SCIENCE_PASS',phase,flush=True)
    (STATE/'validated-runtime.json').write_text(json.dumps(receipt,indent=2))


def flow():
    import asyncio
    from datetime import datetime,timedelta,timezone
    import httpx
    import numpy as np
    import psycopg
    from psycopg.rows import dict_row
    from scipy.io import wavfile
    from starlette.staticfiles import StaticFiles
    import uvicorn
    from ptb_api.main import create_app
    from ptb_api.auth import AuthSettings
    from ptb_api.account_store import hash_token
    from ptb_api.registration import RegistrationService
    from ptb_api.registration_store import PostgresRegistrationStore
    from ptb_api.storage import Storage
    from ptb_api.workspaces import WorkspaceService
    from ptb_worker.store import PostgresJobStore
    from ptb_worker.acoustic_files import AcousticFiles
    from ptb_worker.acoustic_batches import AcousticBatches
    from ptb_worker.executor import run_worker
    from ptb_worker import acoustic_executor
    config=json.loads((STATE/'private.json').read_text())
    storage=Storage(DSN,STATE/'assets',min_free_bytes=10_000_000_000)
    workspace=WorkspaceService(storage);workspace.activate();storage.recover()
    accounts=PostgresRegistrationStore(DSN)
    jobs=PostgresJobStore(DSN,max_running=1,lease_seconds=10)
    files=AcousticFiles(jobs,storage);files.reaper_binary=str(ROOT/'reaper');AcousticBatches(jobs,files)
    origin='http://127.0.0.1:18765'
    class Mail:
        token=None
        def send(self,recipient,subject,body,key):self.token=re.search(r'#verify=([A-Za-z0-9_-]+)',body)[1]
    mail=Mail()
    app=create_app(account_store=accounts,auth_settings=AuthSettings(origin=origin,signing_key=config['signing_key'],allow_insecure_loopback=True),
        job_store=jobs,storage=storage,workspace_service=workspace,registration_service=RegistrationService(accounts,mail,origin))
    app.mount('/server',StaticFiles(directory=ROOT/'frontend/dist',html=True))
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=18765,log_level='error',access_log=False,proxy_headers=False,limit_concurrency=128,timeout_keep_alive=5))
    stop=threading.Event()
    worker=threading.Thread(target=run_worker,args=(jobs,stop,'linux-web'),daemon=True)
    http=threading.Thread(target=server.run,daemon=True)
    report=dict(scope='isolated Linux HTTP/PostgreSQL/real science; no SMTP/public HTTPS',flows=[],stages=[],science_resources=[],host_samples=[])
    collect=acoustic_executor.collect_scientific
    def measured(entry,request,scratch,limits,stop,on_started=None,evidence=None,on_chunk=None):
        evidence={} if evidence is None else evidence
        try:return collect(entry,request,scratch,limits,stop,on_started,evidence,on_chunk)
        finally:report['science_resources'].append(dict(entry=entry,**evidence))
    acoustic_executor.collect_scientific=measured
    def monitor():
        while not stop.wait(.5):
            memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            report['host_samples'].append(dict(time=time.time(),mem_available_kb=int(memory['MemAvailable'].split()[0]),
                loadavg=Path('/proc/loadavg').read_text().split()[:3],cpu=Path('/proc/stat').read_text().splitlines()[0],
                disk_free_bytes=os.statvfs(STATE).f_bavail*os.statvfs(STATE).f_frsize))
    watcher=threading.Thread(target=monitor,daemon=True)
    with np.load(ROOT/'tests/fixtures/m03/EGG-SYN-PCM16.npz') as data:
        samples=np.column_stack([data['load.egg_signal_raw'],data['load.audio_signal']])
    source=io.BytesIO();wavfile.write(source,44100,samples);raw=source.getvalue()
    users=[]
    # Match the browser's exportFontSnapshot contract, including explicit portable
    # resolution. Requested Windows families stay recorded beside actual Linux fonts.
    font=dict(schema_version='font/1',zh='SimSun',latin='Times New Roman',ipa='Doulos SIL',size_px=14,fallback_policy='portable')
    try:
        http.start();worker.start();watcher.start()
        deadline=time.monotonic()+20
        while not server.started:
            assert http.is_alive() and time.monotonic()<deadline;time.sleep(.05)
        with httpx.Client(base_url=origin,trust_env=False,timeout=30) as client:
            assert client.get('/server/').status_code==200
            challenge=client.get('/api/v1/auth/challenge').json()['csrf_token']
            client.headers.update({'Origin':origin,'X-CSRF-Token':challenge})
            name='linux_'+uuid4().hex[:10]
            r=client.post('/api/v1/auth/register',json=dict(username=name,email=name+'@example.test',password='synthetic-password-only'));assert r.is_success,r.text
            deadline=time.monotonic()+5
            while mail.token is None:assert time.monotonic()<deadline;time.sleep(.02)
            r=client.post('/api/v1/auth/verify-email',json=dict(token=mail.token));assert r.is_success,r.text
            r=client.post('/api/v1/auth/login',json=dict(username=name,password='synthetic-password-only'));assert r.is_success,r.text
            session=r.json();owner=session['user']['id']
            client.headers.update({'X-CSRF-Token':session['csrf_token'],'X-PTB-Account':owner})
            lease=client.post('/api/v1/workspaces/open',json={}).json()
            project=client.post('/api/v1/projects',json=dict(name='Linux synthetic')).json()['id']
            r=client.post('/api/v1/uploads',json=dict(project_id=project,name='synthetic.wav',expected_bytes=len(raw),idempotency_key=uuid4().hex));assert r.is_success,r.text
            asset=r.json()
            for offset in range(0,len(raw),262144):
                r=client.put('/api/v1/uploads/'+asset['id']+'/blocks?offset='+str(offset),content=raw[offset:offset+262144],headers={'Content-Type':'application/octet-stream'});assert r.is_success,r.text
            asset=client.post('/api/v1/uploads/'+asset['id']+'/finalize',json={}).json()
            ref=dict(asset_id=asset['id'],sha256=asset['sha256'])
            assert client.get('/api/v1/workspaces/presence').json()['online_accounts']==1
            assert client.get('/api/v1/storage/usage').json()['quota_bytes']==1_000_000_000
            report['flows'].append('register-captured-mail-verify-login-upload-presence-quota')
            for phase in ('lpc','egg','acoustic'):
                body=dict(project_id=project,idempotency_key=uuid4().hex)
                if phase=='acoustic':body.update(operation='acoustic_analysis',inputs=[dict(audio=ref)],config={});route='batches'
                else:body.update(audio=ref,config=dict(roi_end=.05,font=font) if phase=='lpc' else dict(mode='batch',font=font));route=phase
                r=client.post('/api/v1/jobs/'+route+'/create',json=body);assert r.status_code==201,r.text
                if phase=='acoustic':
                    with jobs.transaction(write=False) as tx:
                        job_id=str(tx.execute('SELECT child_job_id FROM ptb_jobs.acoustic_batch_items WHERE batch_id=?',(r.json()['id'],)).fetchone()['child_job_id'])
                else:job_id=r.json()['id']
                deadline=time.monotonic()+90
                while True:
                    result=client.get('/api/v1/jobs/'+job_id).json()
                    if result['state'] not in ('queued','running','cancel_requested'):break
                    assert time.monotonic()<deadline;time.sleep(.1)
                assert result['state']=='succeeded',result
                blobs={}
                for item in result['result_manifest']['files']:
                    r=client.get('/api/v1/assets/'+item['id']+'/content');assert r.status_code==200
                    assert len(r.content)==item['size_bytes'] and hashlib.sha256(r.content).hexdigest()==item['sha256'];blobs[item['name']]=r.content
                if phase=='lpc':
                    meta=json.loads(blobs['lpc.ptb.json']);rate,audio=wavfile.read(io.BytesIO(blobs['lpc_AUDIO.wav']));a,b=meta['selection']['start_sample'],meta['selection']['end_sample']
                    assert rate==44100 and np.array_equal(audio,samples[a:b].astype(float).mean(axis=1))
                elif phase=='egg':
                    import pandas as pd
                    assert pd.read_csv(io.BytesIO(blobs['egg_DATA.csv']))['CQ'].notna().any()
                else:
                    from ptb_api.acoustic_models import AcousticResult
                    value=AcousticResult.model_validate_json(blobs['result.ptb.json'])
                    assert any(b.actual=='native_reaper' for b in value.metadata.backends)
                report['flows'].append(phase+'-actual-science-and-readback')
            assert client.post('/api/v1/workspaces/open',json={}).json()['id']==lease['id']
            r=client.post('/api/v1/auth/logout');assert r.status_code==204
            storage.cleanup();assert storage.usage(owner)['used_bytes']==0
            assert client.get('/api/v1/assets/'+asset['id']+'/content').status_code==401
            report['flows'].append('reconnect-retains-logout-revokes-and-cleans')
        # Synthetic users bypass mail rate limits only during setup, measured traffic is authenticated HTTP.
        from ptb_api.storage_models import UploadInput
        with psycopg.connect(DSN,autocommit=True,row_factory=dict_row) as c:
            for index in range(100):
                owner=str(uuid4());name='load_'+uuid4().hex[:12]
                c.execute('INSERT INTO ptb_accounts.users(id,username,password_hash) VALUES(%s,%s,%s)',(owner,name,'unused'))
                token,csrf=secrets.token_urlsafe(32),secrets.token_urlsafe(32)
                assert accounts.issue_session(accounts.get_user(name),hash_token(token),csrf,datetime.now(timezone.utc)+timedelta(hours=1),'')
                lease=workspace.open(owner,hash_token(token));project=str(workspace.projects(owner,hash_token(token),'capacity')['id'])
                asset=storage.create(owner,UploadInput(project_id=project,name='synthetic.wav',expected_bytes=len(raw),idempotency_key=uuid4().hex))
                for offset in range(0,len(raw),262144):storage.append(owner,asset['id'],offset,raw[offset:offset+262144])
                asset=storage.finalize(owner,asset['id'])
                users.append(dict(owner=owner,token=token,csrf=csrf,project=project,lease=lease,asset=asset['id'],sha=asset['sha256']))
        async def stage(count):
            from ptb_worker.lpc_jobs import submit
            timings=[];statuses={};u=users[0]
            job=submit(jobs,u['owner'],dict(project_id=u['project'],idempotency_key=uuid4().hex,audio=dict(asset_id=u['asset'],sha256=u['sha']),config=dict(roi_end=.05,font=font)))
            async with httpx.AsyncClient(base_url=origin,trust_env=False,timeout=15,limits=httpx.Limits(max_connections=count,max_keepalive_connections=0)) as client:
                async def requests(u):
                    headers={'Cookie':'ptb-dev-session='+u['token'],'X-PTB-Account':u['owner'],'X-CSRF-Token':u['csrf'],'Origin':origin}
                    for step in range(8):
                        start=time.monotonic()
                        try:
                            if step%4==0:r=await client.post('/api/v1/workspaces/heartbeat',headers=headers,json=dict(id=u['lease'].id,generation=u['lease'].generation))
                            else:r=await client.get('/api/v1/'+('jobs?project_id='+u['project'] if step%4==1 else 'assets/'+u['asset']+'/content' if step%4==2 else 'projects'),headers=headers)
                            status=r.status_code
                            if step%4==2 and status==200:assert hashlib.sha256(r.content).hexdigest()==u['sha']
                        except httpx.HTTPError:status=0
                        timings.append(time.monotonic()-start);statuses[str(status)]=statuses.get(str(status),0)+1
                        await asyncio.sleep(.1)
                await asyncio.gather(*(requests(u) for u in users[:count]))
            deadline=time.monotonic()+90
            while True:
                result=jobs.get(u['owner'],job['id'])
                if result['state'] not in ('queued','running','cancel_requested'):break
                assert time.monotonic()<deadline;await asyncio.sleep(.1)
            assert result['state']=='succeeded',result
            p50,p95,p99=np.percentile(timings,[50,95,99]).tolist();error=sum(v for k,v in statuses.items() if k!='200')/len(timings)
            report['stages'].append(dict(users=count,requests=len(timings),statuses=statuses,p50=p50,p95=p95,p99=p99,error_rate=error,science=result['state']))
            return error>.01 or p95>2
        for count in (10,25,50,100):
            if asyncio.run(stage(count)):report['stopped_at_users']=count;break
        stop.set();worker.join(30);assert not worker.is_alive()
        cleanup=[]
        for u in users:
            start=time.monotonic();workspace.close(u['owner'],hash_token(u['token']));storage.cleanup()
            assert storage.usage(u['owner'])['used_bytes']==0;cleanup.append(time.monotonic()-start)
        report['cleanup_p95_seconds']=float(np.percentile(cleanup,95));report['success']=True
    finally:
        stop.set();server.should_exit=True
        if worker.ident:worker.join(30)
        if http.ident:http.join(15)
        if watcher.ident:watcher.join(2)
        acoustic_executor.collect_scientific=collect
        report['owned_threads_stopped']=not http.is_alive() and not worker.is_alive()
        write('flow-capacity.json',report)


def main():
    assert sys.platform=='linux' and pwd.getpwuid(os.getuid()).pw_name=='ptb-staging'
    assert ROOT==Path('/srv/ptb-staging/releases/web-integration-20261016')
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['setup','science','flow']);args=parser.parse_args()
    environment();globals()[args.phase]()


if __name__=='__main__':main()
