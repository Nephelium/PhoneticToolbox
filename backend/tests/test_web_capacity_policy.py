"""Web limits and operator approval use real PostgreSQL transactions."""
from uuid import uuid4
import pytest
from ptb_api.storage_models import UploadInput
from ptb_api.quota import StorageError
from ptb_api.web_capacity import decide, set_quota
from test_web_workspaces_postgres import web, device
from test_p07_policy_postgres import legacy
from web_postgres_support import cluster


def test_default_project_is_atomic_and_scoped_to_workspace(web):
    from concurrent.futures import ThreadPoolExecutor
    e=web;client,key,lease,existing=device(e)
    # Existing projects and renamed defaults are reused, never duplicated.
    assert client.post('/api/v1/workspaces/default-project',json={}).json()['id']==existing
    e.workspaces.close(e.owner,key)
    assert client.post('/api/v1/workspaces/open',json={'fresh':True}).status_code==200
    with ThreadPoolExecutor(max_workers=6) as pool:
        ids=list(pool.map(lambda _:e.workspaces.projects(e.owner,key,ensure_default=True)['id'],range(6)))
    assert len(set(ids))==1 and ids[0]!=existing
    rows=client.get('/api/v1/projects').json()['projects'];assert len(rows)==1 and rows[0]['name']=='默认项目'
    assert client.patch('/api/v1/projects/'+ids[0],json={'name':'已改名'}).status_code==200
    assert client.post('/api/v1/workspaces/default-project',json={}).json()['id']==ids[0]
    bad=client.post('/api/v1/workspaces/default-project',json={},headers={'X-CSRF-Token':'invalid'})
    assert bad.status_code==403


def test_public_load_counts_jobs_and_requires_authentication(web):
    from ptb_api.web_status import PublicLoad
    e=web;client,_,_,project=device(e)
    r=client.post('/api/v1/jobs',json=dict(project_id=project,idempotency_key=uuid4().hex,operation='pipeline_check',config={'sample_count':4096,'seed':0}))
    assert r.status_code==201,r.text
    load=PublicLoad(e.workspaces,e.jobs,probe=lambda:(None,None)).sample()
    assert load.waiting_tasks==1 and load.running_tasks==0 and load.task_slots==e.jobs.max_running
    response=client.get('/api/v1/workspaces/presence');assert response.status_code==200
    assert response.json()['server']['waiting_tasks']==1
    assert response.headers['cache-control']=='no-store'
    client.cookies.clear();assert client.get('/api/v1/workspaces/presence').status_code==401


def test_default_quota_reservation_and_input_limit(web):
    e=web;client,_,_,project=device(e)
    assert client.get('/api/v1/storage/usage').json()['quota_bytes']==1_000_000_000
    assert client.post('/api/v1/uploads',json=dict(project_id=project,name='large.mp4',expected_bytes=200_000_000,idempotency_key=uuid4().hex)).status_code==413
    # Input and output reservations share the same account balance without writing a huge fixture.
    with e.storage._locked() as c:
        e.storage._create(c,e.owner,UploadInput(project_id=project,name='result',expected_bytes=950_000_000,idempotency_key=uuid4().hex),kind='result')
    r=client.post('/api/v1/uploads',json=dict(project_id=project,name='next.wav',expected_bytes=60_000_000,idempotency_key=uuid4().hex))
    assert r.status_code==413 and r.json()['detail']=='quota_exceeded'
    assert client.get('/api/v1/storage/usage').json()['available_bytes']==50_000_000


def test_presence_counts_accounts_not_tabs_or_recovery_grace(web):
    e=web;a,_,lease,_=device(e);b,_,other,_=device(e)
    assert a.get('/api/v1/workspaces/presence').json()['online_accounts']==1
    e.conn.execute('UPDATE ptb_web.workspaces SET last_seen=last_seen-91,expires_at=expires_at-91 WHERE owner_id=%s',(e.owner,))
    assert a.get('/api/v1/workspaces/presence').json()['online_accounts']==0
    assert a.post('/api/v1/workspaces/heartbeat',json=dict(id=lease['id'],generation=lease['generation'])).status_code==200
    assert b.get('/api/v1/workspaces/presence').json()['online_accounts']==1
    a.post('/api/v1/auth/logout')
    assert b.get('/api/v1/workspaces/presence').json()['online_accounts']==0


def test_manual_capacity_grant_and_duplicate_request(web):
    e=web;client,_,_,project=device(e)
    body=dict(requested_bytes=1_500_000_000,reason='Synthetic capacity verification')
    r=client.post('/api/v1/workspaces/capacity-request',json=body)
    assert r.status_code==200,r.text
    assert client.post('/api/v1/workspaces/capacity-request',json=body).json()['id']==r.json()['id']
    assert client.get('/api/v1/storage/usage').json()['quota_bytes']==1_000_000_000
    assert client.post('/api/v1/workspaces/capacity-request/'+r.json()['id']+'/approve',json={}).status_code in (404,405)
    decide(e.workspaces,r.json()['id'],granted_bytes=1_500_000_000,note='synthetic approval')
    assert client.get('/api/v1/storage/usage').json()['quota_bytes']==1_500_000_000
    with e.storage._locked() as c:
        e.storage._create(c,e.owner,UploadInput(project_id=project,name='result',expected_bytes=1_000_000_000,idempotency_key=uuid4().hex),kind='result')
    assert client.get('/api/v1/storage/usage').json()['available_bytes']==500_000_000
    with pytest.raises(StorageError):decide(e.workspaces,r.json()['id'],granted_bytes=1_500_000_000)


def test_admin_default_and_individual_override_preserve_data(web):
    e=web;client,_,_,project=device(e)
    value=e.storage.create(e.owner,UploadInput(project_id=project,name='source',expected_bytes=4,idempotency_key=uuid4().hex))
    e.storage.append(e.owner,value['id'],0,b'test');e.storage.finalize(e.owner,value['id'])
    set_quota(e.workspaces,500_000_000)
    assert e.storage.usage(e.owner)['quota_bytes']==500_000_000
    set_quota(e.workspaces,2_000_000_000,e.owner)
    set_quota(e.workspaces,750_000_000)
    assert e.storage.usage(e.owner)['quota_bytes']==2_000_000_000
    assert e.storage.read_block(e.owner,value['id'],0,4)==b'test'
    request=client.post('/api/v1/workspaces/capacity-request',json=dict(requested_bytes=3_000_000_000,reason='synthetic',duration='today')).json()
    result=decide(e.workspaces,request['id'],granted_bytes=3_000_000_000)
    from datetime import timedelta,timezone
    assert result.expires_at.astimezone(timezone(timedelta(hours=8))).hour==0
    e.conn.execute("UPDATE ptb_storage.quota_accounts SET web_override_expires_at=clock_timestamp()-interval '1 second' WHERE owner_id=%s",(e.owner,))
    assert e.storage.usage(e.owner)['quota_bytes']==2_000_000_000


def test_approval_reply_duration_override_and_expiry(web):
    e=web;client,_,_,project=device(e)
    request=client.post('/api/v1/workspaces/capacity-request',json=dict(requested_bytes=2_000_000_000,reason='synthetic',duration='30d')).json()
    result=decide(e.workspaces,request['id'],granted_bytes=1_500_000_000,note='先批准7天',duration='7d')
    assert result.duration=='30d' and result.granted_duration=='7d' and result.decision_note=='先批准7天'
    with e.storage._locked() as c:
        a=e.storage._create(c,e.owner,UploadInput(project_id=project,name='result',expected_bytes=1_000_000_000,idempotency_key=uuid4().hex),kind='result')
        e.storage._create(c,e.owner,UploadInput(project_id=project,name='other',expected_bytes=400_000_000,idempotency_key=uuid4().hex),kind='result')
    e.conn.execute("UPDATE ptb_storage.quota_accounts SET web_override_expires_at=clock_timestamp()-interval '1 second' WHERE owner_id=%s",(e.owner,))
    usage=client.get('/api/v1/storage/usage').json()
    assert usage['quota_bytes']==1_000_000_000 and usage['over_quota'] and usage['reserved_bytes']==1_400_000_000
    # An already reserved write can settle, but no new reservation is accepted.
    with e.storage._locked() as c:
        e.storage._append(c,e.owner,a['id'],0,b'test')
    r=client.post('/api/v1/uploads',json=dict(project_id=project,name='next.wav',expected_bytes=1,idempotency_key=uuid4().hex))
    assert r.status_code==413
