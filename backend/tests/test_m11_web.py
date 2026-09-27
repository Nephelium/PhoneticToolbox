"""Real authenticated ASGI/PG queue and quota. No remote execution claim.

Uses the existing fresh-cluster fixture, never an existing account database.
"""
from uuid import uuid4
import json
import hashlib
from test_p07_policy_postgres import cluster,legacy,migrate


def test_mfa_web_owned_inputs_wait_without_local_fallback(legacy,tmp_path,monkeypatch):
    from argon2 import PasswordHasher
    from fastapi.testclient import TestClient
    from ptb_api.account_store import PostgresAccountStore
    from ptb_api.auth import AuthSettings
    from ptb_api.main import create_app
    from ptb_worker.store import PostgresJobStore
    from ptb_worker.acoustic_files import AcousticFiles
    e=legacy;migrate(e)
    e.conn.execute('UPDATE ptb_accounts.users SET password_hash=%s WHERE id=%s',(PasswordHasher().hash('synthetic-test-only'),e.owner))
    second=str(uuid4());e.conn.execute("INSERT INTO ptb_accounts.users(id,username,password_hash) VALUES(%s,'second',%s)",(second,PasswordHasher().hash('synthetic-test-only')))
    monkeypatch.setenv('PTB_M11_COMPONENT_ROOT',str(tmp_path))
    # Registered metadata fixture tests routing only, never masquerades as a node.
    (tmp_path/'registry.json').write_text(json.dumps(dict(schema='m11-registry/1',runtimes=[dict(id='test-mfa338',fingerprint='b'*64,version='3.3.8',platform='linux',arch='x86_64')],models=[dict(id='test-model',name='test',model_sha256='c'*64,dictionary_sha256='d'*64,validated_runtime='test-mfa338')])),encoding='utf8')
    jobs=PostgresJobStore(e.dsn,lease_seconds=60);files=AcousticFiles(jobs,e.storage)
    origin='https://m11.test'
    app=create_app(account_store=PostgresAccountStore(e.dsn),auth_settings=AuthSettings(origin=origin,signing_key='m11-synthetic-only-'*4),job_store=jobs,storage=e.storage)
    def login(client,user,owner):
        csrf=client.get('/api/v1/auth/challenge').json()['csrf_token']
        result=client.post('/api/v1/auth/login',headers={'Origin':origin,'X-CSRF-Token':csrf},json=dict(username=user,password='synthetic-test-only'))
        assert result.status_code==200,result.text
        return {'Origin':origin,'X-CSRF-Token':result.json()['csrf_token'],'X-PTB-Account':owner}
    with TestClient(app,base_url=origin) as client,TestClient(app,base_url=origin) as other:
        headers=login(client,'synthetic',e.owner);other_headers=login(other,'second',second)
        def upload(name,raw):
            created=client.post('/api/v1/uploads',headers=headers,json=dict(project_id=e.project,name=name,expected_bytes=len(raw),idempotency_key=uuid4().hex));assert created.status_code==201,created.text
            asset=created.json();assert client.put('/api/v1/uploads/'+asset['id']+'/blocks?offset=0',headers=headers|{'Content-Type':'application/octet-stream'},content=raw).status_code==200
            ready=client.post('/api/v1/uploads/'+asset['id']+'/finalize',headers=headers,json=dict(sha256=hashlib.sha256(raw).hexdigest()));assert ready.status_code==200,ready.text
            return dict(asset_id=asset['id'],sha256=ready.json()['sha256'])
        audio=upload('公开.wav',b'public routing fixture');text=upload('公开.lab',b'a')
        body=dict(project_id=e.project,idempotency_key=uuid4().hex,runtime_id='test-mfa338',model_id='test-model',corpus=[dict(name='公开.wav',audio=audio,transcript=text)])
        denied=other.post('/api/v1/jobs/m11/create',headers=other_headers,json=body);assert denied.status_code==404,denied.text
        result=client.post('/api/v1/jobs/m11/create',headers=headers,json=body);assert result.status_code==201,result.text
        job=result.json();assert job['state']=='queued' and job['waiting_reason']=='m11_waiting_verified_node'
        assert client.post('/api/v1/jobs/m11/create',headers=headers,json=body).json()['id']==job['id']
        assert jobs.claim('local-worker-must-not-run-mfa') is None
        restored=PostgresJobStore(e.dsn);AcousticFiles(restored,e.storage)
        assert restored.get(e.owner,job['id'])['state']=='queued'
        assert other.get('/api/v1/jobs/'+job['id'],headers=other_headers).status_code==404
        assert other.get('/api/v1/assets/'+audio['asset_id']+'/content',headers=other_headers).status_code==404
        usage=client.get('/api/v1/storage/usage',headers=headers).json()
        assert usage['quota_bytes']==1_000_000_000 and usage['retention_seconds']==259200
        assert usage['used_bytes']==len(b'public routing fixture')+1
        assert client.get('/api/v1/jobs/m11/catalog',headers=headers).json()['execution_available'] is False
        assert client.post('/api/v1/jobs/m11/component',headers=headers,json=dict(action='check',runtime='forbidden',model='forbidden',dictionary='forbidden')).status_code==404
        assert client.post('/api/v1/jobs/'+job['id']+'/cancel',headers=headers).json()['state']=='cancelled'
