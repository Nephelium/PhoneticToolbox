"""Real transactions against a fresh, synthetic, owned PostgreSQL cluster."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from secrets import token_urlsafe
from uuid import uuid4
import pytest
import psycopg
from fastapi.testclient import TestClient
from ptb_api.account_store import PostgresAccountStore,hash_token
from ptb_api.auth import AuthSettings
from ptb_api.main import create_app
from ptb_api.storage_models import UploadInput
from ptb_api.job_models import FileJobInput
from ptb_api.quota import StorageError
from ptb_api.workspaces import WorkspaceService,WorkspaceHeartbeat
from ptb_worker.files import FilePipeline
from ptb_worker.store import PostgresJobStore
from test_p07_policy_postgres import legacy,migrate,MIGRATIONS
from web_postgres_support import cluster

ORIGIN='https://workspace.test'


@pytest.fixture
def web(legacy):
    e=legacy;migrate(e)
    for name in ('008_web_identity.sql','009_web_workspaces.sql','010_web_capacity.sql'):
        e.conn.execute((MIGRATIONS/name).read_text('utf-8'))
    e.accounts=PostgresAccountStore(e.dsn)
    e.workspaces=WorkspaceService(e.storage);e.workspaces.activate()
    e.jobs=PostgresJobStore(e.dsn);e.files=FilePipeline(e.jobs,e.storage)
    e.app=create_app(account_store=e.accounts,auth_settings=AuthSettings(origin=ORIGIN,signing_key='workspace-test-'*4),
        job_store=e.jobs,storage=e.storage,workspace_service=e.workspaces)
    e.clients=[]
    yield e
    for client in e.clients:client.close()


def device(e):
    raw=token_urlsafe(32);csrf=token_urlsafe(32)
    user=e.accounts.get_user('synthetic')
    assert e.accounts.issue_session(user,hash_token(raw),csrf,datetime.now(timezone.utc)+timedelta(hours=8),'')
    client=TestClient(e.app,base_url=ORIGIN)
    client.cookies.set('__Host-ptb-session',raw,domain='workspace.test',path='/')
    client.headers.update({'Origin':ORIGIN,'X-CSRF-Token':csrf,'X-PTB-Account':e.owner})
    e.clients.append(client)
    lease=client.post('/api/v1/workspaces/open',json={'fresh':False})
    assert lease.status_code==200,lease.text
    assert lease.headers['X-PTB-Web-Use']=='1'
    project=client.post('/api/v1/projects',json={'name':'合成研究'})
    assert project.status_code==201,project.text
    return client,hash_token(raw),lease.json(),project.json()['id']


def asset(e,project):
    body=UploadInput(project_id=project,name='synthetic.bin',expected_bytes=4,idempotency_key=uuid4().hex)
    row=e.storage.create(e.owner,body)
    e.storage.append(e.owner,row['id'],0,b'test')
    return e.storage.finalize(e.owner,row['id'])


def expire(e,lease,seconds=1801):
    e.conn.execute('UPDATE ptb_web.workspaces SET last_seen=last_seen-%s,expires_at=expires_at-%s WHERE id=%s',(seconds,seconds,lease['id']))


def test_same_session_refresh_multitab_and_other_device_isolation(web):
    e=web;a,key,lease,project=device(e);b,_,other,other_project=device(e)
    assert lease['id']!=other['id'] and lease['expires_at']-lease['last_seen']==1800
    refreshed=a.post('/api/v1/workspaces/open',json={})
    assert refreshed.json()==lease
    assert refreshed.headers['X-PTB-Web-Use']=='0'
    assert a.get('/api/v1/projects').json()['projects'][0]['id']==project
    assert b.get('/api/v1/projects/'+project).status_code==404
    assert b.post('/api/v1/uploads',json=dict(project_id=project,name='bad',idempotency_key=uuid4().hex)).status_code==404
    assert b.post('/api/v1/jobs',json=dict(project_id=project,idempotency_key=uuid4().hex)).status_code==404
    assert a.get('/api/v1/projects/'+e.project).status_code==404  # legacy project not adopted
    assert a.patch('/api/v1/projects/'+project,json={'name':'刷新后项目'}).status_code==200
    value=asset(e,project)
    assert value['expires_at']==lease['expires_at']
    assert b.get('/api/v1/assets/'+value['id']).status_code==404
    e.storage.read_block(e.owner,value['id'],0,4)
    assert e.workspaces.current(e.owner,key)['last_seen']==lease['last_seen']
    beat=a.post('/api/v1/workspaces/heartbeat',json={'id':lease['id'],'generation':1})
    assert beat.status_code==200,beat.text
    assert e.storage.metadata(e.owner,value['id'])['expires_at']==beat.json()['expires_at']
    assert a.get('/api/v1/storage/usage').json()['retention_seconds']==1800
    assert a.get('/api/v1/projects').headers['cache-control']=='no-store'


def test_expiry_denies_old_data_and_requires_explicit_fresh_start(web):
    e=web;a,key,lease,project=device(e);value=asset(e,project)
    expire(e,lease)
    assert a.post('/api/v1/workspaces/heartbeat',json={'id':lease['id'],'generation':1}).status_code==410
    assert a.get('/api/v1/projects').status_code==410
    assert a.get('/api/v1/assets/'+value['id']).status_code==410
    assert a.post('/api/v1/workspaces/open',json={}).status_code==410
    new=a.post('/api/v1/workspaces/open',json={'fresh':True})
    assert new.status_code==200 and new.json()['id']!=lease['id']
    assert a.get('/api/v1/projects').json()=={'projects':[]}
    assert a.get('/api/v1/projects/'+project).status_code==404
    assert a.post('/api/v1/workspaces/heartbeat',json={'id':lease['id'],'generation':1}).status_code==409
    e.storage.cleanup()
    assert not e.storage._path(value['id']).exists()
    assert e.conn.execute('SELECT 1 FROM ptb_accounts.projects WHERE id=%s',(project,)).fetchone() is None
    assert e.accounts.get_user('synthetic') is not None


def test_logout_fences_worker_immediately_and_cleanup_retries_physical_failures(web,monkeypatch):
    e=web;a,key,lease,project=device(e);b,_,_,other_project=device(e)
    value=asset(e,project);other=asset(e,other_project)
    job=e.jobs.submit(e.owner,FileJobInput(project_id=project,idempotency_key=uuid4().hex,operation='storage_check'))
    claim=e.files.claim('synthetic-worker');identity=(job['id'],'synthetic-worker',claim['generation'])
    output=e.files.output(identity,'result.bin','result',4);e.files.write(identity,output['id'],0,b'data');e.files.seal(identity,output['id'])
    assert a.post('/api/v1/auth/logout').status_code==204
    assert e.accounts.session(key) is None
    with pytest.raises(StorageError):e.files.complete(identity)
    assert e.jobs.get(e.owner,job['id'])['state']=='cancelled'
    original=Path.unlink
    def denied(path,*args,**kwargs):
        if path==e.storage._path(value['id']):raise PermissionError('synthetic lock')
        return original(path,*args,**kwargs)
    with monkeypatch.context() as m:
        m.setattr(Path,'unlink',denied);e.storage.cleanup()
        assert e.storage.usage(e.owner)['used_bytes']>=8
        assert e.conn.execute('SELECT state FROM ptb_web.workspaces WHERE id=%s',(lease['id'],)).fetchone()['state']=='closed'
    e.storage.cleanup()
    assert e.storage.usage(e.owner)['used_bytes']==4
    assert e.storage.metadata(e.owner,other['id'])['state']=='ready'
    assert b.get('/api/v1/projects/'+other_project).status_code==200
    for table in ('ptb_jobs.jobs','ptb_storage.assets','ptb_web.workspace_projects'):
        assert e.conn.execute('SELECT count(*) AS n FROM '+table+' WHERE project_id=%s',(project,)).fetchone()['n']==0


def test_completed_results_capped_and_late_publication_rejected_in_database(web):
    e=web;a,key,lease,project=device(e)
    job=e.jobs.submit(e.owner,FileJobInput(project_id=project,idempotency_key=uuid4().hex,operation='storage_check'))
    claim=e.files.claim('synthetic-worker');identity=(job['id'],'synthetic-worker',claim['generation'])
    result=e.files.output(identity,'result.bin','result',4);e.files.write(identity,result['id'],0,b'data');e.files.seal(identity,result['id'])
    manifest=e.files.complete(identity)
    assert manifest['files'][0]['expires_at']==lease['expires_at']
    assert e.workspaces.current(e.owner,key)['last_seen']==lease['last_seen']
    expire(e,lease)
    with pytest.raises(psycopg.errors.RaiseException,match='workspace_expired'):
        e.conn.execute("UPDATE ptb_jobs.jobs SET state='succeeded' WHERE id=%s",(job['id'],))
    e.storage.recover()
    assert e.storage.ready and e.storage.usage(e.owner)['used_bytes']==0
    assert e.conn.execute('SELECT state FROM ptb_web.workspaces WHERE id=%s',(lease['id'],)).fetchone()['state']=='cleaned'


def test_concurrent_heartbeat_and_close_never_revives_workspace(web):
    e=web;_,key,lease,project=device(e)
    def beat():
        try:return e.workspaces.heartbeat(e.owner,key,WorkspaceHeartbeat(id=lease['id'],generation=1))
        except StorageError:return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=[pool.submit(beat),pool.submit(e.workspaces.close,e.owner,key)]
        for result in results:result.result()
    assert e.conn.execute('SELECT state FROM ptb_web.workspaces WHERE id=%s',(lease['id'],)).fetchone()['state']=='closed'
    with pytest.raises(StorageError):e.workspaces.current(e.owner,key)


def test_web_preview_memory_is_released_only_for_ended_asset(web):
    from types import SimpleNamespace
    e=web;_,key,lease,project=device(e);value=asset(e,project);released=[]
    preview=SimpleNamespace(owner=(e.owner,value['id']),token='synthetic',release=lambda *identity:released.append(identity))
    e.workspaces.release_expired_preview(preview);assert not released
    e.workspaces.close(e.owner,key);e.workspaces.release_expired_preview(preview)
    assert released==[((e.owner,value['id']),'synthetic')]
