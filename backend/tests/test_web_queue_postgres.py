"""Hosted queue admission uses real scheduling transactions, not UI counters."""
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import pytest
from ptb_api.job_models import FileJobInput
from ptb_worker.acoustic_files import AcousticFiles
from ptb_worker.acoustic_batches import AcousticBatches
from ptb_worker.store import JobError
from test_web_workspaces_postgres import web, device, asset
from test_p07_policy_postgres import legacy
from web_postgres_support import cluster


def submit(e, project, key=None):
    return e.jobs.submit(e.owner, FileJobInput(project_id=project,
        idempotency_key=key or uuid4().hex, operation='storage_check'))


def test_concurrent_account_limit_idempotency_cancel_and_slot(web):
    e=web
    client,_,_,project=device(e)
    def attempt(_):
        try:return submit(e,project)
        except JobError as exc:return exc.code
    with ThreadPoolExecutor(max_workers=10) as pool:
        results=list(pool.map(attempt,range(16)))
    assert sum(isinstance(r,dict) for r in results)==8
    assert results.count('account_queue_full')==8
    key=uuid4().hex
    e.jobs.cancel(e.owner,next(r['id'] for r in results if isinstance(r,dict)))
    job=submit(e,project,key)
    assert submit(e,project,key)['id']==job['id']
    response=client.post('/api/v1/jobs',json=dict(project_id=project,idempotency_key=uuid4().hex,operation='pipeline_check'))
    assert response.status_code==429 and response.headers['retry-after']=='5'
    first=e.jobs.claim('one')
    assert first and e.jobs.claim('two') is None  # existing per-owner execution slot


def test_batch_reserves_uncreated_children_and_global_limit(web):
    e=web
    client,_,_,project=device(e)
    AcousticBatches(e.jobs,AcousticFiles(e.jobs,e.storage))
    refs=[]
    for n in range(8):
        value=asset(e,project)
        e.conn.execute("UPDATE ptb_storage.assets SET name=%s WHERE id=%s",(f'synthetic-{n}.wav',value['id']))
        refs.append(dict(asset_id=value['id'],sha256=value['sha256']))
    body=dict(project_id=project,operation='acoustic_analysis',idempotency_key=uuid4().hex,
        inputs=[{'audio':ref} for ref in refs],config={})
    batch=e.jobs.batches.submit(e.owner,body)
    assert e.jobs.batches.submit(e.owner,body)['id']==batch['id']
    with pytest.raises(JobError,match='account_queue_full'):submit(e,project)
    e.jobs.batches.cancel(e.owner,batch['id'])
    submit(e,project)
    # Fill the shared bound with synthetic jobs of distinct accounts in this DB.
    for n in range(16):
        owner,other_project=uuid4(),uuid4()
        e.conn.execute('INSERT INTO ptb_accounts.users(id,username,password_hash) VALUES(%s,%s,%s)',(owner,'queue_'+str(n),'unused'))
        e.conn.execute('INSERT INTO ptb_accounts.projects(id,owner_id,name) VALUES(%s,%s,%s)',(other_project,owner,'synthetic'))
        e.conn.execute('''INSERT INTO ptb_jobs.jobs(id,owner_id,project_id,idempotency_key,request_hash,snapshot,state,deadline,created_at,updated_at)
            SELECT gen_random_uuid(),%s,%s,gen_random_uuid()::text,repeat('a',64),'{}','queued',9999999999,0,0 FROM generate_series(1,%s)''',
            (owner,other_project,7 if n==15 else 8))
    with pytest.raises(JobError,match='server_queue_full'):submit(e,project)
