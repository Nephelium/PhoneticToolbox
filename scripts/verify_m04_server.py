"""M04-C existing dedicated PostgreSQL + real ASGI auth/storage/worker. No DDL."""
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import threading
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
from fastapi.testclient import TestClient
from ptb_api.account_store import PostgresAccountStore
from ptb_api.auth import AuthSettings
from ptb_api.main import create_app
from ptb_api.storage import Storage
from ptb_api.storage_models import UploadInput
from ptb_api.quota import QUOTA_BYTES
from ptb_worker.store import PostgresJobStore
from ptb_worker.acoustic_files import AcousticFiles
from ptb_worker.acoustic_batches import AcousticBatches
from ptb_worker.acoustic_executor import execute_acoustic_claim
from run_m01_validation import owned_postgres

ROOT=Path(__file__).resolve().parents[1]


def verify(config,out):
    account=PostgresAccountStore(config['dsn']);storage=Storage(config['dsn'],ROOT/'output/validation/p07/storage')
    jobs=PostgresJobStore(config['dsn'])
    with jobs.transaction(write=False) as tx:
        assert not tx.execute("SELECT 1 FROM {jobs} WHERE state IN ('queued','running','cancel_requested')").fetchone()
    storage.recover();files=AcousticFiles(jobs,storage);AcousticBatches(jobs,files)
    with jobs.transaction(write=False) as tx:old=[dict(r) for r in tx.execute('SELECT * FROM {jobs}').fetchall()]
    origin='http://127.0.0.1:5179'
    app=create_app('server',account_store=account,auth_settings=AuthSettings(origin,secrets.token_urlsafe(48),True),job_store=jobs,storage=storage)
    people=[];stream=io.BytesIO();wavfile.write(stream,16000,np.random.default_rng(4).normal(size=8000));raw=stream.getvalue()
    os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
    with ExitStack() as stack:
        for _ in range(2):
            username='m04_'+uuid4().hex[:12];password=secrets.token_urlsafe(32)
            user=account.create_user(username,password);owner=str(user['id'])
            project=str(account.create_project(owner,'LPC 服务验收')['id'])
            client=stack.enter_context(TestClient(app,base_url=origin))
            csrf=client.get('/api/v1/auth/challenge').json()['csrf_token']
            response=client.post('/api/v1/auth/login',json=dict(username=username,password=password),headers={'Origin':origin,'x-csrf-token':csrf})
            assert response.status_code==200
            headers={'Origin':origin,'x-csrf-token':response.json()['csrf_token']}
            asset=storage.create(owner,UploadInput(project_id=project,name='LPC ɑ̃˥.wav',expected_bytes=len(raw),idempotency_key=uuid4().hex))
            storage.append(owner,asset['id'],0,raw);storage.finalize(owner,asset['id'],hashlib.sha256(raw).hexdigest())
            people.append(dict(owner=owner,project=project,client=client,headers=headers,ref=dict(asset_id=asset['id'],sha256=hashlib.sha256(raw).hexdigest())))
        a,b=people
        def post(person,**changes):
            body=dict(project_id=person['project'],idempotency_key=uuid4().hex,audio=person['ref'],config={'roi_end':.1})|changes
            return person['client'].post('/api/v1/jobs/lpc/create',json=body,headers=person['headers'])
        assert post(b,audio=a['ref']).status_code==409
        assert post(a,project_id=b['project']).status_code==404
        job=post(a);assert job.status_code==201,job.text;job=job.json()
        assert b['client'].get('/api/v1/jobs/'+job['id']).status_code==404
        assert b['client'].post('/api/v1/jobs/'+job['id']+'/cancel',headers=b['headers']).status_code==404
        claim=jobs.claim('m04-server');assert claim['id']==job['id']
        execute_acoustic_claim(jobs,claim,'m04-server',threading.Event())
        done=a['client'].get('/api/v1/jobs/'+job['id']).json();assert done['state']=='succeeded',done
        for file in done['result_manifest']['files']:
            path='/api/v1/assets/'+file['id']+'/content'
            assert b['client'].get(path).status_code==404
            response=a['client'].get(path);assert response.status_code==200
            assert hashlib.sha256(response.content).hexdigest()==file['sha256']
            (out/file['name']).write_bytes(response.content)
        # Hold remaining quota with an owned reservation, then execute a task.
        usage=storage.usage(a['owner'])
        reservation=storage.create(a['owner'],UploadInput(project_id=a['project'],name='quota-test.bin',
            expected_bytes=QUOTA_BYTES-usage['used_bytes']-usage['reserved_bytes'],idempotency_key=uuid4().hex))
        try:
            quota=post(a);assert quota.status_code==201
            claim=jobs.claim('m04-server');assert claim['id']==quota.json()['id']
            execute_acoustic_claim(jobs,claim,'m04-server',threading.Event())
            failed=jobs.get(a['owner'],claim['id']);assert failed['state']=='failed' and failed['error_code']=='quota_exceeded',failed
        finally:storage.delete(a['owner'],reservation['id'])
        # Controlled expiry touches this run's output only, not a seven-day wait.
        expired=done['result_manifest']['files'][0]['id']
        with storage._locked() as conn:
            conn.execute('UPDATE ptb_storage.assets SET expires_at=0 WHERE id=%s AND owner_id=%s',(expired,a['owner']))
        assert a['client'].get('/api/v1/assets/'+expired+'/content').status_code==410
        storage.delete(a['owner'],expired);assert not storage._path(expired).exists()
        with storage._locked() as conn:
            assert not conn.execute("SELECT 1 FROM ptb_storage.assets WHERE owner_id=%s AND kind='temporary' AND state!='deleted'",(a['owner'],)).fetchone()
        assert storage.usage(a['owner'])['reserved_bytes']==0
        with jobs.transaction(write=False) as tx:
            for row in old:assert dict(tx.execute('SELECT * FROM {jobs} WHERE id=?',(row['id'],)).fetchone())==row
    return dict(success=True,authenticated_accounts=2,old_job_rows_preserved=True,outputs_readback=3,
        cross_owner_denied=True,quota_failure_reclaimed=True,controlled_expiry_deleted=True,temporary_files_remaining=0,
        transport='FastAPI TestClient ASGI with actual PG/storage/Windows child; browser interaction is D')


def main():
    out=ROOT/'output/validation/m04-server'/uuid4().hex;out.mkdir(parents=True)
    report=dict(success=False,schema_applied=[])
    try:
        with owned_postgres(out) as config:report.update(verify(config,out))
        report['owned_postgres_stopped']=True
    finally:
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(out/'report.json')


if __name__=='__main__':main()
