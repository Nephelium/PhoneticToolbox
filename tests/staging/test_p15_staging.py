"""P15 deployment adapter tests; doubles are not host/PG/TLS acceptance."""
import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/p15_staging'))
from host import validate_config, SiteBoundary, GateError, check_manifest
from ptb_api.main import create_app
from ptb_api.auth import AuthSettings
from account_double import MemoryAccountStore


def config():
    return dict(schema='p15-staging/1', origin='https://staging.example.org',
                bind='127.0.0.1', port=18765, unix_user='ptb-staging',
                release_root='/srv/ptb-staging/releases/test',
                frontend='/srv/ptb-staging/releases/test/frontend/dist',
                storage_root='/var/lib/ptb-staging/assets',
                private_config='/etc/ptb-staging/private.json',
                runtime_profile='/srv/ptb-staging/releases/test/runtime.json',
                validation_receipt='/srv/ptb-staging/releases/test/receipt.json',
                release_manifest='/srv/ptb-staging/releases/test/release.json',
                control_file='/var/lib/ptb-staging/control.json',
                min_free_bytes=10737418240, max_running=1, lease_seconds=10,
                resource_profile='server-small', allowed_operations=[], remote_enabled=False)


def test_closed_config_valid():
    assert validate_config(config())['allowed_operations'] == []


@pytest.mark.parametrize('changes', [
    {'origin':'http://staging.example.org'}, {'origin':'https://x.test/path'},
    {'origin':'https://user:secret@x.test'}, {'bind':'0.0.0.0'},
    {'max_running':2}, {'resource_profile':'desktop-local'},
    {'remote_enabled':True}, {'allowed_operations':['pitch_manipulation']},
    {'allowed_operations':['phonology_induction']}, {'allowed_operations':['archive_zip']},
    {'storage_root':'/srv/ptb-staging/releases/test/frontend/dist/private'},
    {'min_free_bytes':0}, {'port':True}, {'origin':'https://REPLACE_DOMAIN'},
    {'private_config':'relative'}, {'unexpected':'ignored'}])
def test_config_rejects_unsafe_or_unimplemented(changes):
    with pytest.raises(GateError): validate_config(config() | changes)


def build_site(tmp_path, mode='maintenance'):
    control=tmp_path/'control.json'; control.write_text(json.dumps({'mode':mode}))
    accounts=MemoryAccountStore(); accounts.create_user('alice','test-password-only')
    app=create_app(account_store=accounts,auth_settings=AuthSettings(
        origin='https://staging.example.org',signing_key='test-only-'*8))
    return SiteBoundary(app,'staging.example.org',control), control, accounts


def login(client):
    csrf=client.get('/api/v1/auth/challenge').json()['csrf_token']
    result=client.post('/api/v1/auth/login',json={'username':'alice','password':'test-password-only'},
        headers={'Origin':'https://staging.example.org','X-CSRF-Token':csrf})
    assert result.status_code==200
    return {'Origin':'https://staging.example.org','X-CSRF-Token':result.json()['csrf_token']}


def test_live_maintenance_and_bad_control_fail_closed(tmp_path):
    site, control, _=build_site(tmp_path)
    with TestClient(site,base_url='https://staging.example.org') as c:
        h=login(c)
        assert c.post('/api/v1/projects',json={'name':'test'},headers=h).status_code==503
        control.write_text('{invalid')
        assert c.get('/api/v1/projects').status_code==503
        control.write_text('{"mode":"open"}')
        assert c.post('/api/v1/projects',json={'name':'test'},headers=h).status_code==201
        control.write_text('{"mode":"drain"}')
        assert c.get('/api/v1/projects').status_code==200
        assert c.post('/api/v1/uploads',json={},headers=h).status_code==503
        assert c.post('/api/v1/auth/logout',headers=h).status_code==204


def test_host_https_csrf_and_cookies(tmp_path):
    site, _, _=build_site(tmp_path,'open')
    with TestClient(site,base_url='https://staging.example.org') as c:
        assert c.get('/api/v1/health',headers={'Host':'evil.test'}).status_code==403
        assert c.get('/api/v1/health',headers={'Host':'staging.example.org','X-Forwarded-Host':'evil.test'}).status_code==200
        h=login(c)
        assert c.post('/api/v1/projects',json={'name':'test'},headers=h|{'Origin':'https://evil.test'}).status_code==403
        assert c.post('/api/v1/projects',json={'name':'test'},headers={'Origin':h['Origin']}).status_code==403
        assert c.get('/api/v1/health').headers['server-timing'].startswith('app;dur=')
    with TestClient(site,base_url='http://staging.example.org') as c:
        assert c.get('/api/v1/health').status_code==400


def test_manifest_detects_replaced_bytes_and_missing_coverage(tmp_path):
    root=tmp_path/'release';root.mkdir()
    (root/'a.py').write_text('one')
    import hashlib
    manifest={'schema':'p15-release/1','files':{'a.py':hashlib.sha256(b'one').hexdigest()}}
    check_manifest(root,manifest,required={'a.py'})
    with pytest.raises(GateError): check_manifest(root,manifest,required={'a.py','b.py'})
    (root/'a.py').write_text('two')
    with pytest.raises(GateError): check_manifest(root,manifest,required={'a.py'})
    with pytest.raises(GateError): check_manifest(root,{'schema':'p15-release/1','files':{'../escape':'0'*64}},required=set())


def test_deployment_gate_blocks_unverified_and_retry_paths(tmp_path):
    site,_,_=build_site(tmp_path,'open')
    with TestClient(site,base_url='https://staging.example.org') as c:
        for path in ('/api/v1/jobs','/api/v1/jobs/m08/create','/api/v1/jobs/m14/create',
                     '/api/v1/jobs/spec2wav/create','/api/v1/jobs/any/retry','/api/v1/worker/poll'):
            assert c.post(path,json={}).status_code==503


def test_synthetic_driver_roundtrip_with_explicit_storage_double(tmp_path):
    """Test the driver with real API routes + memory doubles. NOT real TLS/PG."""
    from datetime import datetime,timezone
    from email.utils import format_datetime
    import hashlib,time
    from uuid import uuid4
    from ptb_api.quota import StorageError
    from verify_site import authenticated_checks,Checks,synthetic_wav

    class StorageDouble:
        ready=True
        def __init__(self):self.rows={};self.data={}
        def usage(self,owner):
            return dict(policy_version=2,retention_seconds=259200,over_quota=False,
                        quota_bytes=1000000000,used_bytes=0,reserved_bytes=0,available_bytes=1000000000,
                        frozen=False,ready=True)
        def create(self,owner,body):
            key=str(uuid4());now=time.time()
            self.rows[key]=dict(id=key,owner=owner,project_id=str(body.project_id),name=body.name,
                kind='input',state='uploading',size_bytes=0,reserved_bytes=body.expected_bytes,
                expected_bytes=body.expected_bytes,sha256=None,created_at=now,expires_at=now+86400,
                error_code=None,policy_version=2)
            self.data[key]=b'';return self.rows[key]
        def metadata(self,owner,key):
            r=self.rows.get(str(key))
            if not r or r['owner']!=owner:raise StorageError('asset_not_found',404)
            return r
        def append(self,owner,key,offset,data):
            r=self.metadata(owner,key);key=str(key)
            assert offset==len(self.data[key]);self.data[key]+=data
            r['size_bytes']=len(self.data[key]);return r
        def finalize(self,owner,key,expected_hash):
            r=self.metadata(owner,key);r.update(state='ready',reserved_bytes=0,
                sha256=hashlib.sha256(self.data[str(key)]).hexdigest(),expires_at=time.time()+259200)
            assert r['sha256']==expected_hash;return r
        def read_block(self,owner,key,offset,size):
            self.metadata(owner,key);return self.data[str(key)][offset:offset+size]
        def delete(self,owner,key):
            r=self.metadata(owner,key);r['state']='deleted';return r

    accounts=MemoryAccountStore()
    credentials=[{'username':'alice','password':'test-a-only'},{'username':'bob','password':'test-b-only'}]
    for item in credentials:accounts.create_user(item['username'],item['password'])
    app=create_app(account_store=accounts,auth_settings=AuthSettings(origin='https://staging.example.org',signing_key='test-only-'*8),storage=StorageDouble())
    @app.middleware('http')
    async def date_header(request,call_next):
        r=await call_next(request);r.headers['Date']=format_datetime(datetime.now(timezone.utc),usegmt=True);return r
    control=tmp_path/'control.json';control.write_text('{"mode":"open"}')
    site=SiteBoundary(app,'staging.example.org',control)
    checks=Checks()
    with TestClient(site,base_url='https://staging.example.org') as a,TestClient(site,base_url='https://staging.example.org') as b:
        result=authenticated_checks(a,b,credentials,checks)
    assert result['input_bytes']==3244
    assert result['input_sha256']==hashlib.sha256(synthetic_wav()).hexdigest()
    serialized=json.dumps(checks.samples)
    assert all(item['password'] not in serialized and item['username'] not in serialized for item in credentials)


def test_policy_preflight_uses_readonly_and_rejects_old_database(monkeypatch):
    import psycopg
    from host import database_state
    class Connection:
        def __init__(self):self.sql=[];self.version=1
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def execute(self,sql):self.sql.append(sql);self.last=sql;return self
        def fetchone(self):
            if 'FROM ptb_storage.state' in self.last:return {'instance_id':'instance','policy_version':self.version,'frozen':False}
            return {'invalid_quota':0,'over_quota':1}
    c=Connection();monkeypatch.setattr(psycopg,'connect',lambda *a,**k:c)
    with pytest.raises(GateError,match='migration_required'):database_state('private','instance')
    assert c.sql[0]=='SET TRANSACTION READ ONLY'
    assert all(not any(w in sql.upper().split() for w in ('UPDATE','DELETE','ALTER','CREATE')) for sql in c.sql)
    c.version=2
    assert database_state('private','instance')['over_quota_accounts']==1
    with pytest.raises(GateError,match='instance_mismatch'):database_state('private','wrong')


def test_templates_render_without_installing_and_reject_injection(tmp_path):
    from render import render
    templates=Path(__file__).resolve().parents[2]/'deployment/p15-staging'
    names=render(templates,tmp_path/'review','staging.example.org','p15-test-1')
    assert 'ptb-staging@.service' in names
    assert json.loads((tmp_path/'review/control.json').read_text())=={'mode':'maintenance'}
    with pytest.raises(FileExistsError):render(templates,tmp_path/'review','staging.example.org','p15-test-1')
    for domain in ('x.test;shutdown','x.test\nserver {}','../x.test'):
        with pytest.raises(ValueError):render(templates,tmp_path/'other',domain,'p15-test-1')


def test_real_proxy_middleware_only_trusts_loopback(tmp_path):
    import asyncio
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
    site,_,_=build_site(tmp_path,'open')
    wrapped=ProxyHeadersMiddleware(site,trusted_hosts='127.0.0.1')
    async def request(peer):
        sent=[]
        scope={'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'GET',
            'scheme':'http','path':'/api/v1/health','raw_path':b'/api/v1/health','query_string':b'',
            'root_path':'','server':('127.0.0.1',18765),'client':(peer,2222),
            'headers':[(b'host',b'staging.example.org'),(b'x-forwarded-proto',b'https'),
                       (b'x-forwarded-for',b'203.0.113.9')]}
        import anyio
        async def receive():await anyio.sleep_forever()
        async def send(message):sent.append(message)
        await wrapped(scope,receive,send)
        return next(m['status'] for m in sent if m['type']=='http.response.start')
    assert asyncio.run(request('127.0.0.1'))==200
    assert asyncio.run(request('203.0.113.10'))==400


def test_lpc_driver_checks_contract_and_rejects_corrupt_result():
    from verify_site import lpc_check,Checks,CheckFailed
    from ptb_api.lpc_models import LpcRequest
    from uuid import uuid4
    import hashlib,httpx
    project,asset,job=[str(uuid4()) for _ in range(3)]
    payload=b'synthetic-result'
    results=[{'id':str(uuid4()),'name':name,'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
             for name in ('lpc.ptb.json','lpc_SPECTRUM.png','lpc_AUDIO.wav')]
    class Client:
        def __init__(self,other=False,corrupt=False):self.other=other;self.corrupt=corrupt
        def request(self,method,path,**kw):
            if self.other:return httpx.Response(404,json={'detail':'not_found'})
            if path.endswith('/fonts'):return httpx.Response(200,json={'available':True})
            if path.endswith('/create'):
                LpcRequest.model_validate(kw['json'])
                return httpx.Response(201,json={'id':job})
            if '/content' in path:return httpx.Response(200,content=b'corrupt' if self.corrupt else payload)
            if '?' in path:return httpx.Response(200,json={'jobs':[{'id':job}]})
            return httpx.Response(200,json={'state':'succeeded','result_manifest':{'complete':True,'policy_version':2,'files':results}})
    checks=Checks()
    lpc_check(Client(),Client(other=True),{}, {},project,asset,'0'*64,checks)
    assert checks.owned['synthetic_job_id']==job
    with pytest.raises(CheckFailed,match='integrity'):
        lpc_check(Client(corrupt=True),Client(other=True),{}, {},project,asset,'0'*64,Checks())


def test_unreviewed_queue_holds_without_claiming():
    from host import queue_is_supported
    from contextlib import contextmanager
    class Store:
        @contextmanager
        def transaction(self,write=True):
            assert write is False
            yield self
        def execute(self,sql):
            assert sql.startswith('SELECT DISTINCT')
            return [{'operation':'lpc_analysis'},{'operation':'phonology_induction'}]
        def claim(self,*args):raise AssertionError('Must not claim an unsupported task')
    assert not queue_is_supported(Store(),{'lpc_analysis'})
