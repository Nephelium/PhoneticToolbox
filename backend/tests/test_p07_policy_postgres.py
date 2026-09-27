"""Fresh-cluster Q21-Q25 integration, NEVER an existing/service database.

Opt in with PTB_POLICY_FRESH_PG=1. Starts only a newly initdb'd private cluster;
uses bundled PG, random loopback port, random DB/root per test. Keeps evidence
directories, stops only its own cluster. No DSN/config from user environment.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import patch
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import tempfile
import threading
import time

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
import pytest
from ptb_api.storage import Storage
from ptb_api.storage_models import UploadInput
from ptb_api.quota import StorageError
from ptb_api.job_models import FileJobInput, JobView
from ptb_worker.files import FilePipeline
from ptb_worker.store import PostgresJobStore
from ptb_worker.executor import execute_claim

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = ROOT/'backend/migrations'


@pytest.fixture(scope='module')
def cluster():
    if os.environ.get('PTB_POLICY_FRESH_PG') != '1':
        pytest.skip('requires opt-in for a fresh synthetic PostgreSQL cluster')
    binaries = ROOT/'.venv/postgresql-17.11-3/pgsql/bin'
    if os.name != 'nt' or not (binaries/'initdb.exe').exists():
        pytest.skip('bundled Windows PostgreSQL required for this isolated runner')
    evidence = ROOT/'output/validation/p07-policy'
    evidence.mkdir(parents=True,exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='fresh-',dir=evidence))
    data = directory/'pgdata'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
    kwargs = dict(stderr=subprocess.STDOUT,text=True,encoding='utf-8',
        creationflags=subprocess.CREATE_NO_WINDOW)
    def run(args):
        # Windows postgres can inherit pg_ctl pipe handles. A real log file
        # avoids communicate waiting for EOF from the long-running child.
        with (directory/'commands.log').open('a',encoding='utf-8') as log:
            result = subprocess.run([str(binaries/args[0]),*args[1:]],stdout=log,**kwargs,timeout=60)
        assert result.returncode == 0, (directory/'commands.log').read_text('utf-8')
    run(['initdb.exe','-D',str(data),'-U','policy_test','-A','trust','--no-locale','-E','UTF8'])
    try:
        run(['pg_ctl.exe','-D',str(data),'-l',str(directory/'postgres.log'),'-w','start',
            '-o',f'-h 127.0.0.1 -p {port} -c max_connections=20 -c shared_buffers=16MB'])
        yield dict(host='127.0.0.1',port=port,user='policy_test',dbname='postgres'),directory
    finally:
        if (data/'postmaster.pid').exists():
            run(['pg_ctl.exe','-D',str(data),'-w','stop','-m','fast'])
        assert not (data/'postmaster.pid').exists()


@pytest.fixture
def legacy(cluster):
    config,directory = cluster
    dbname = 'policy_'+uuid4().hex
    with psycopg.connect(**config,autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(dbname)))
    dsn = psycopg.conninfo.make_conninfo(**(config|{'dbname':dbname}))
    conn = psycopg.connect(dsn,autocommit=True,row_factory=dict_row)
    for name in ('001_accounts.sql','002_jobs.sql','003_storage.sql','004_job_assets.sql','005_acoustic_batches.sql'):
        conn.execute((MIGRATIONS/name).read_text('utf-8'))
    owner,project,instance = uuid4(),uuid4(),uuid4()
    conn.execute("INSERT INTO ptb_accounts.users(id,username,password_hash) VALUES(%s,'synthetic','unused')",(owner,))
    conn.execute("INSERT INTO ptb_accounts.projects(id,owner_id,name) VALUES(%s,%s,'synthetic')",(project,owner))
    conn.execute('INSERT INTO ptb_storage.state(version,instance_id,frozen) VALUES(1,%s,false)',(instance,))
    conn.execute('INSERT INTO ptb_storage.quota_accounts(owner_id) VALUES(%s)',(owner,))
    root = directory/dbname
    root.mkdir()
    (root/'.ptb-storage.json').write_text(json.dumps({'instance_id':str(instance)}),'utf-8')
    (root/'.ptb-storage.lock').write_bytes(b'0')
    storage = Storage(dsn,root,min_free_bytes=0)
    storage.ready = True
    env = SimpleNamespace(conn=conn,storage=storage,owner=str(owner),project=str(project),dsn=dsn,root=root)
    yield env
    conn.close()


def migrate(env):
    env.conn.execute((MIGRATIONS/'006_storage_policy.sql').read_text('utf-8'))


def old_asset(env, data=b'legacy', reserved=0, expires=None, state='ready'):
    asset_id = uuid4()
    expires = expires or time.time()+604800
    env.conn.execute('''INSERT INTO ptb_storage.assets(id,owner_id,project_id,name,state,
        idempotency_key,request_hash,expected_bytes,size_bytes,reserved_bytes,sha256,created_at,expires_at)
        VALUES(%s,%s,%s,'旧文件.bin',%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
        (asset_id,env.owner,env.project,state,uuid4().hex,'a'*64,len(data)+reserved,len(data),reserved,
         hashlib.sha256(data).hexdigest() if state=='ready' else None,time.time(),expires))
    env.conn.execute('UPDATE ptb_storage.quota_accounts SET used_bytes=used_bytes+%s,reserved_bytes=reserved_bytes+%s WHERE owner_id=%s',
        (len(data),reserved,env.owner))
    env.storage._path(asset_id).write_bytes(data)
    return str(asset_id),expires


def upload(env,data=b'new',expected=None):
    asset = env.storage.create(env.owner,UploadInput(project_id=env.project,name='合成.bin',
        expected_bytes=len(data) if expected is None else expected,idempotency_key=uuid4().hex))
    if data: env.storage.append(env.owner,asset['id'],0,data)
    return env.storage.finalize(env.owner,asset['id'])


def test_unmigrated_database_read_delete_but_no_new_writes(legacy):
    e = legacy
    asset,expires = old_asset(e)
    assert e.storage.usage(e.owner)['policy_version'] == 1
    assert e.storage.usage(e.owner)['quota_bytes'] == 5_000_000_000
    assert e.storage.read_block(e.owner,asset,0,10) == b'legacy'
    assert e.storage.metadata(e.owner,asset)['expires_at'] == expires
    with pytest.raises(StorageError,match='storage_policy_migration_required'): upload(e)
    assert e.storage.delete(e.owner,asset)['state'] == 'deleted'


def test_readonly_preflight_and_migration_preserve_expiry_reservations_and_old_manifest(legacy):
    e = legacy
    asset,expires = old_asset(e)
    pending,_ = old_asset(e,b'',reserved=1_000_000_001,state='uploading')
    before = e.conn.execute('SELECT * FROM ptb_storage.assets ORDER BY id').fetchall()
    old_manifest=json.dumps(dict(kind='managed_files',complete=True,core_version='3.0.0a1',
        files=[dict(id=asset,name='旧文件.bin',kind='result',size_bytes=6,sha256=hashlib.sha256(b'legacy').hexdigest(),expires_at=expires)]))
    snapshot=json.dumps(dict(operation='storage_check',config={'max_output_bytes':5_000_000_000}))
    job_id=str(uuid4())
    e.conn.execute('''INSERT INTO ptb_jobs.jobs(id,owner_id,project_id,idempotency_key,request_hash,
        snapshot,state,deadline,created_at,updated_at,result_manifest,progress)
        VALUES(%s,%s,%s,%s,%s,%s,'succeeded',%s,0,1,%s,1)''',
        (job_id,e.owner,e.project,uuid4().hex,'b'*64,snapshot,expires,old_manifest))
    spec = importlib.util.spec_from_file_location('preflight',ROOT/'scripts/p07_policy_preflight.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with psycopg.connect(e.dsn,row_factory=dict_row) as read:
        report = module.inspect(read, e.root)
        assert report['ready_for_review'] and report['counts']['over_quota_accounts'] == 1
        assert report['disk']['instance_matches'] and all(v==[1] for v in report['schema_versions'].values())
        assert read.execute('SHOW transaction_read_only').fetchone()['transaction_read_only'] == 'on'
    migrate(e)
    assert module.metadata_fingerprints(e.conn) == report['metadata_fingerprints']
    after = e.conn.execute('SELECT * FROM ptb_storage.assets ORDER BY id').fetchall()
    assert [{k:v for k,v in row.items() if k!='policy_version'} for row in after] == before
    assert all(row['policy_version']==1 for row in after)
    old_job=PostgresJobStore(e.dsn).get(e.owner,job_id)
    assert JobView.model_validate(old_job).result_manifest.files[0].expires_at==expires
    persisted=e.conn.execute('SELECT snapshot,result_manifest FROM ptb_jobs.jobs WHERE id=%s',(job_id,)).fetchone()
    assert persisted==dict(snapshot=snapshot,result_manifest=old_manifest)
    usage = e.storage.usage(e.owner)
    assert usage['quota_bytes']==1_000_000_000 and usage['over_quota'] and usage['available_bytes']==0
    assert usage['reserved_bytes']==1_000_000_001
    assert e.storage.read_block(e.owner,asset,0,10)==b'legacy'
    assert e.storage.metadata(e.owner,asset)['expires_at']==expires
    with pytest.raises(StorageError,match='quota_exceeded'): upload(e)
    with pytest.raises(StorageError,match='quota_exceeded'): e.storage.append(e.owner,pending,0,b'x')
    assert e.storage.delete(e.owner,pending)['state']=='deleted'
    assert e.storage.usage(e.owner)['used_bytes']==6
    assert not e.storage.usage(e.owner)['over_quota']
    assert upload(e)['policy_version']==2


def test_last_byte_concurrent_reservations_and_database_growth_guard(legacy):
    e = legacy
    old_asset(e,b'',reserved=999_999_999,state='uploading')
    migrate(e)
    def contender(_):
        try:
            e.storage.create(e.owner,UploadInput(project_id=e.project,name='one.bin',expected_bytes=1,idempotency_key=uuid4().hex))
            return 'ok'
        except StorageError as error: return error.code
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes=list(pool.map(contender,range(4)))
    assert outcomes.count('ok')==1 and outcomes.count('quota_exceeded')==3
    assert e.storage.usage(e.owner)['reserved_bytes']==1_000_000_000
    with pytest.raises(psycopg.errors.CheckViolation):
        e.conn.execute('UPDATE ptb_storage.quota_accounts SET reserved_bytes=reserved_bytes+1 WHERE owner_id=%s',(e.owner,))


def test_upload_three_days_exact_expiry_and_download_no_renewal(legacy):
    e = legacy
    migrate(e)
    with patch.object(e.storage,'_now',return_value=1000.):
        asset=upload(e,b'new-content')
        assert asset['expires_at']==260200.
        assert e.storage.read_block(e.owner,asset['id'],0,20)==b'new-content'
        assert e.storage.metadata(e.owner,asset['id'])['expires_at']==260200.
    with patch.object(e.storage,'_now',return_value=260199.999):
        assert e.storage.read_block(e.owner,asset['id'],0,20)==b'new-content'
    with patch.object(e.storage,'_now',return_value=260200.):
        with pytest.raises(StorageError,match='asset_expired'): e.storage.read_block(e.owner,asset['id'],0,20)


def test_delete_failure_keeps_used_and_reserved_then_retries(legacy):
    e = legacy
    asset,_=old_asset(e,b'written',reserved=1_000_000_001,state='uploading')
    migrate(e)
    before=e.storage.usage(e.owner)
    with patch.object(Path,'unlink',side_effect=PermissionError('synthetic denial')):
        assert e.storage.delete(e.owner,asset)['state']=='delete_failed'
    after=e.storage.usage(e.owner)
    assert (after['used_bytes'],after['reserved_bytes'])==(before['used_bytes'],before['reserved_bytes'])
    assert e.storage._path(asset).read_bytes()==b'written'
    assert e.storage.delete(e.owner,asset)['state']=='deleted'
    assert e.storage.usage(e.owner)['used_bytes']==e.storage.usage(e.owner)['reserved_bytes']==0


def test_inflight_recovery_settles_without_growing_and_preserves_deadline(legacy):
    e = legacy
    asset,deadline=old_asset(e,b'',reserved=1_000_000_010,state='uploading')
    # Crash after fsync before settlement: this byte is covered by reservations.
    e.storage._path(asset).write_bytes(b'x')
    migrate(e)
    e.storage.recover()
    usage=e.storage.usage(e.owner)
    assert usage['used_bytes']==1 and usage['reserved_bytes']==1_000_000_009
    assert e.storage.list(e.owner,e.project)[0]['expires_at']==deadline
    with pytest.raises(StorageError,match='quota_exceeded'): e.storage.append(e.owner,asset,1,b'y')


def pipeline(e):
    jobs=PostgresJobStore(e.dsn,lease_seconds=60)
    files=FilePipeline(jobs,e.storage)
    def submit(operation='storage_check',inputs=()):
        return jobs.submit(e.owner,FileJobInput(project_id=e.project,idempotency_key=uuid4().hex,
            operation=operation,config={'inputs':list(inputs),'probe_files':1,'probe_bytes':10}))
    def claim(job):
        row=jobs.claim('policy-worker')
        assert row['id']==job['id']
        return row,(row['id'],row['worker_id'],row['generation'])
    return jobs,files,submit,claim


def test_real_generated_and_zip_results_with_old_input_expiry(legacy):
    e = legacy
    source,deadline=old_asset(e,b'zip-input',expires=time.time()+3600)
    migrate(e)
    jobs,files,submit,claim=pipeline(e)
    for operation in ('storage_check','archive_zip'):
        job=submit(operation,[source]);row,identity=claim(job)
        before=time.time()
        execute_claim(jobs,row,row['worker_id'],threading.Event())
        after=time.time()
        result=jobs.get(e.owner,job['id'])
        assert result['state']=='succeeded',result
        item=result['result_manifest']['files'][0]
        assert JobView.model_validate(result).state=='succeeded'
        if operation=='storage_check': assert before+259200<=item['expires_at']<=after+259200
        else: assert item['expires_at']==deadline
        assert e.storage.read_block(e.owner,item['id'],0,256)
    assert e.storage.metadata(e.owner,source)['expires_at']==deadline


def test_input_expiry_and_stale_worker_cannot_publish(legacy):
    e=legacy
    source,_=old_asset(e)
    migrate(e)
    jobs,files,submit,claim=pipeline(e)
    job=submit('storage_check',[source]);row,identity=claim(job)
    output=files.output(identity,'result.bin','result',1)
    files.write(identity,output['id'],0,b'x');files.seal(identity,output['id'])
    with pytest.raises(StorageError,match='stale_worker'):
        files.complete((identity[0],identity[1],identity[2]+1))
    e.conn.execute('UPDATE ptb_storage.assets SET expires_at=%s WHERE id=%s',(time.time()-1,source))
    with pytest.raises(StorageError,match='input_unavailable'): files.complete(identity)
    assert jobs.get(e.owner,job['id'])['result_manifest'] is None
    e.storage.cleanup()
    assert jobs.get(e.owner,job['id'])['state']=='cancel_requested'
    files.fail(identity,'input_unavailable')
    assert jobs.get(e.owner,job['id'])['state']=='cancelled'
    assert not e.storage._path(output['id']).exists()


def test_inflight_completed_bytes_can_finalize_and_disk_full_does_not_reserve(legacy):
    e=legacy
    complete,_=old_asset(e,b'complete',state='uploading')
    old_asset(e,b'',reserved=1_000_000_001,state='uploading')
    migrate(e)
    before=time.time()
    result=e.storage.finalize(e.owner,complete)
    assert before+259200<=result['expires_at']<=time.time()+259200
    assert result['policy_version']==2
    used=e.storage.usage(e.owner)
    with patch('ptb_api.storage.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
        with pytest.raises(StorageError,match='disk_space_low'): upload(e)
    assert e.storage.usage(e.owner)==used


def test_over_quota_http_login_usage_download_and_delete_are_still_available(legacy):
    from argon2 import PasswordHasher
    from fastapi.testclient import TestClient
    from ptb_api.account_store import PostgresAccountStore
    from ptb_api.auth import AuthSettings
    from ptb_api.main import create_app
    e=legacy
    asset,expires=old_asset(e)
    old_asset(e,b'',reserved=1_000_000_001,state='uploading')
    e.conn.execute('UPDATE ptb_accounts.users SET password_hash=%s WHERE id=%s',
        (PasswordHasher().hash('synthetic-test-only'),e.owner))
    migrate(e)
    origin='https://policy.test'
    app=create_app(account_store=PostgresAccountStore(e.dsn),
        auth_settings=AuthSettings(origin=origin,signing_key='synthetic-policy-key-'*4),storage=e.storage)
    with TestClient(app,base_url=origin) as client:
        csrf=client.get('/api/v1/auth/challenge').json()['csrf_token']
        session=client.post('/api/v1/auth/login',headers={'Origin':origin,'X-CSRF-Token':csrf},
            json={'username':'synthetic','password':'synthetic-test-only'})
        assert session.status_code==200
        headers={'Origin':origin,'X-CSRF-Token':session.json()['csrf_token'],'X-PTB-Account':e.owner}
        usage=client.get('/api/v1/storage/usage',headers=headers).json()
        assert usage['over_quota'] and usage['policy_version']==2 and usage['retention_seconds']==259200
        response=client.get('/api/v1/assets/'+asset+'/content?expected_account='+e.owner,headers=headers)
        assert response.status_code==200 and response.content==b'legacy'
        assert e.storage.metadata(e.owner,asset)['expires_at']==expires
        body=UploadInput(project_id=e.project,name='new',expected_bytes=1,idempotency_key=uuid4().hex).model_dump(mode='json')
        assert client.post('/api/v1/uploads',json=body,headers=headers).status_code==413
        assert client.delete('/api/v1/assets/'+asset,headers=headers).json()['state']=='deleted'


def test_migration_rechecks_ledger_and_rolls_back_atomically(legacy):
    e=legacy
    e.conn.execute('UPDATE ptb_storage.quota_accounts SET used_bytes=1 WHERE owner_id=%s',(e.owner,))
    with pytest.raises(psycopg.errors.RaiseException,match='policy_preflight_mismatch'): migrate(e)
    e.conn.execute('ROLLBACK')
    assert e.conn.execute('SELECT quota_bytes FROM ptb_storage.quota_accounts').fetchone()['quota_bytes']==5_000_000_000
    assert 'policy_version' not in e.conn.execute('SELECT * FROM ptb_storage.state').fetchone()
