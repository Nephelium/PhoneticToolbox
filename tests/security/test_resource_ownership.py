"""P05 route-level owner checks; PostgreSQL adapter requires separate integration run."""
from fastapi.testclient import TestClient
from ptb_api.main import create_app
from ptb_api.auth import AuthSettings
from account_double import MemoryAccountStore

ORIGIN='https://research.test'

def test_two_accounts_project_ownership_and_csrf():
    store=MemoryAccountStore()
    for name in ('alice','bob'):store.create_user(name,'long-password-for-test')
    app=create_app(account_store=store,auth_settings=AuthSettings(origin=ORIGIN,signing_key='test-only-'*8))
    with TestClient(app,base_url=ORIGIN) as a, TestClient(app,base_url=ORIGIN) as b:
        for client,name in ((a,'alice'),(b,'bob')):
            token=client.get('/api/v1/auth/challenge').json()['csrf_token']
            r=client.post('/api/v1/auth/login',json={'username':name,'password':'long-password-for-test'},headers={'Origin':ORIGIN,'X-CSRF-Token':token})
            assert r.status_code==200
        ah={'Origin':ORIGIN,'X-CSRF-Token':a.get('/api/v1/auth/me').json()['csrf_token']}
        bh={'Origin':ORIGIN,'X-CSRF-Token':b.get('/api/v1/auth/me').json()['csrf_token']}
        created=a.post('/api/v1/projects',json={'name':'声门事件研究'},headers=ah)
        assert created.status_code==201
        project=created.json()['id']
        assert a.get('/api/v1/projects').json()['projects'][0]['id']==project
        assert b.get('/api/v1/projects').json()['projects']==[]
        assert b.get('/api/v1/projects/'+project).status_code==404
        assert b.patch('/api/v1/projects/'+project,json={'name':'stolen'},headers=bh).status_code==404
        assert a.post('/api/v1/projects',json={'name':'x','owner_id':store.users['bob']['id']},headers=ah).status_code==422
        assert a.patch('/api/v1/projects/'+project,json={'name':'wrong'},headers=bh).status_code==403
        assert a.post('/api/v1/projects',json={'name':'x'},headers={**ah,'Origin':'https://evil.test'}).status_code==403
        assert a.get('/api/v1/projects/'+project).json()['name']=='声门事件研究'
        assert a.get('/api/v1/projects').headers['cache-control']=='no-store'
        assert a.post('/api/v1/projects',json={'name':'   '},headers=ah).status_code==422
        assert a.patch('/api/v1/projects/not-a-uuid',json={'name':'x'},headers=ah).status_code==422

def test_local_and_unconfigured_server_do_not_offer_fake_accounts():
    for mode,expected in [('local',404),('server',503)]:
        with TestClient(create_app(mode)) as client:
            assert client.get('/api/v1/auth/me').status_code==expected
            assert client.get('/api/v1/projects').status_code==expected
            assert client.post('/api/v1/jobs',json={}).status_code==404
