"""Local HTTP load staircase, not a claim about the 2-core Linux server.

Synthetic identities bypass email/login rate limits during setup; measured calls
use real cookies, ownership, CSRF, PostgreSQL and storage. Stop on overload.
"""
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
from secrets import token_urlsafe
import socket
import threading
import time
from uuid import uuid4

import httpx
import numpy as np
from scipy.io import wavfile
import uvicorn
import os

from ptb_api.account_store import hash_token
from ptb_api.auth import AuthSettings
from ptb_api.main import create_app
from ptb_api.storage_models import UploadInput
from ptb_worker.acoustic_batches import AcousticBatches
from ptb_worker.acoustic_files import AcousticFiles
from ptb_worker.executor import run_worker
from ptb_worker.lpc_jobs import submit
from test_web_workspaces_postgres import web
from test_p07_policy_postgres import legacy, ROOT
from web_postgres_support import cluster, artifacts


def test_local_http_staircase_with_real_science(web, monkeypatch):
    e=web
    e.jobs.max_running=1
    AcousticBatches(e.jobs, AcousticFiles(e.jobs,e.storage))
    monkeypatch.setenv('PTB_EGG_PYTHON',str(ROOT/'.venv/m03-compatible/python.exe'))
    with artifacts.verification_run('web-launch','capacity-local',
            '10/25/50/100 virtual users, actual loopback HTTP and isolated PG, one LPC worker; stop at >1% errors or p95 >2 s') as (out,scratch):
        import io
        source=io.BytesIO()
        wavfile.write(source,16000,np.random.default_rng(42).normal(0,.1,8000))
        raw=source.getvalue()
        sock=socket.socket();sock.bind(('127.0.0.1',0))
        origin='http://127.0.0.1:'+str(sock.getsockname()[1])
        app=create_app(account_store=e.accounts,auth_settings=AuthSettings(origin=origin,
            signing_key='capacity-synthetic-'*3,allow_insecure_loopback=True),job_store=e.jobs,
            storage=e.storage,workspace_service=e.workspaces)
        server=uvicorn.Server(uvicorn.Config(app,log_level='error',access_log=False,
            proxy_headers=False,limit_concurrency=128,timeout_keep_alive=5))
        http=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True)
        stop=threading.Event();worker=threading.Thread(target=run_worker,args=(e.jobs,stop,'capacity-worker'),daemon=True)
        report=dict(scope='Windows local infrastructure staircase; NOT Linux server capacity',
            connection_limit=128,postgres_max_connections=20,stages=[],science=[],cleanup_seconds=[])
        report['host']=dict(logical_cpus=os.cpu_count())
        report['science_resources']=[]
        from ptb_worker import acoustic_executor
        original_collect=acoustic_executor.collect_scientific
        def measured(entry, request, scratch, limits, stop, on_started=None, evidence=None, on_chunk=None):
            evidence={} if evidence is None else evidence
            try:return original_collect(entry,request,scratch,limits,stop,on_started,evidence,on_chunk)
            finally:report['science_resources'].append(dict(entry=entry,**evidence))
        monkeypatch.setattr(acoustic_executor,'collect_scientific',measured)
        users=[]
        try:
            # Only synthetic fixtures in the newly created database, no existing users.
            for index in range(100):
                owner=str(uuid4());name='load_'+str(index)
                e.conn.execute('INSERT INTO ptb_accounts.users(id,username,password_hash) VALUES(%s,%s,%s)',(owner,name,'unused'))
                token,csrf=token_urlsafe(32),token_urlsafe(32)
                assert e.accounts.issue_session(e.accounts.get_user(name),hash_token(token),csrf,datetime.now(timezone.utc)+timedelta(hours=1),'')
                lease=e.workspaces.open(owner,hash_token(token),False)
                project=e.workspaces.projects(owner,hash_token(token),'capacity')
                project=str(project['id'])
                a=e.storage.create(owner,UploadInput(project_id=project,name='synthetic.wav',expected_bytes=len(raw),idempotency_key=uuid4().hex))
                e.storage.append(owner,a['id'],0,raw);a=e.storage.finalize(owner,a['id'])
                users.append(dict(owner=owner,token=token,csrf=csrf,project=project,lease=lease,asset=a['id'],sha=a['sha256']))
            http.start();worker.start()
            until=time.monotonic()+15
            while not server.started:
                assert http.is_alive() and time.monotonic()<until
                time.sleep(.01)

            async def stage(count):
                timings=[];statuses={};samples=[]
                # A task genuinely computes while the measured traffic is active.
                u=users[0]
                job=submit(e.jobs,u['owner'],dict(project_id=u['project'],idempotency_key=uuid4().hex,
                    audio=dict(asset_id=u['asset'],sha256=u['sha']),config=dict(roi_end=.05)))
                async with httpx.AsyncClient(base_url=origin,trust_env=False,timeout=15,
                        limits=httpx.Limits(max_connections=count,max_keepalive_connections=0)) as client:
                    async def user_flow(u):
                        headers={'Cookie':'ptb-dev-session='+u['token'],'X-PTB-Account':u['owner'],
                            'X-CSRF-Token':u['csrf'],'Origin':origin}
                        for step in range(8):
                            started=time.perf_counter()
                            try:
                                if step%4==0:
                                    lease=u['lease']
                                    r=await client.post('/api/v1/workspaces/heartbeat',headers=headers,json=dict(id=lease.id,generation=lease.generation))
                                elif step%4==1:r=await client.get('/api/v1/jobs?project_id='+u['project'],headers=headers)
                                elif step%4==2:r=await client.get('/api/v1/assets/'+u['asset']+'/content',headers=headers)
                                else:r=await client.get('/api/v1/projects',headers=headers)
                                status=r.status_code
                                if step%4==2 and status==200:assert hashlib.sha256(r.content).hexdigest()==u['sha']
                            except httpx.HTTPError:status=0
                            elapsed=time.perf_counter()-started
                            timings.append(elapsed);statuses[str(status)]=statuses.get(str(status),0)+1
                            samples.append(dict(route=step%4,status=status,seconds=elapsed))
                            await asyncio.sleep(.1)
                    started=time.perf_counter();cpu_started=time.process_time()
                    await asyncio.gather(*(user_flow(u) for u in users[:count]))
                    elapsed=time.perf_counter()-started
                    cpu_seconds=time.process_time()-cpu_started
                deadline=time.monotonic()+45
                while True:
                    result=e.jobs.get(u['owner'],job['id'])
                    if result['state'] not in ('queued','running','cancel_requested'):break
                    assert time.monotonic()<deadline
                    await asyncio.sleep(.1)
                report['science'].append(dict(users=count,state=result['state'],error=result['error_code']))
                assert result['state']=='succeeded',result
                pct=np.percentile(timings,[50,95,99]).tolist()
                errors=sum(n for status,n in statuses.items() if status!='200')/len(timings)
                value=dict(users=count,requests=len(timings),elapsed_seconds=elapsed,p50=pct[0],p95=pct[1],p99=pct[2],statuses=statuses,error_rate=errors,
                    api_cpu_seconds=cpu_seconds,resource_scope='API process CPU and separate native job memory peak; PostgreSQL not included')
                artifacts.write_json(out/f'samples-{count}.json',samples)
                report['stages'].append(value)
                return errors>.01 or pct[1]>2
            for count in (10,25,50,100):
                if asyncio.run(stage(count)):
                    report['stopped_at_users']=count
                    break
            # Never turn the staircase stop into a false all-pass capacity claim.
            assert report['stages'][0]['error_rate']==0,report['stages'][0]
            stop.set();worker.join(30);assert not worker.is_alive()
            for u in users:
                started=time.perf_counter()
                e.workspaces.close(u['owner'],hash_token(u['token']))
                e.storage.cleanup()
                assert e.storage.usage(u['owner'])['used_bytes']==0
                report['cleanup_seconds'].append(time.perf_counter()-started)
            report['cleanup_p95_seconds']=float(np.percentile(report['cleanup_seconds'],95))
        finally:
            stop.set()
            if worker.ident:worker.join(30)
            server.should_exit=True
            if http.ident:http.join(15)
            sock.close()
            artifacts.write_json(out/'report.json',report)
            assert not worker.is_alive() and not http.is_alive()
