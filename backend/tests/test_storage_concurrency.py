"""A5 real PostgreSQL regressions. Fresh local cluster, no existing database."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import threading
import time
from uuid import uuid4
import pytest
from ptb_api.postgres_pool import connection, pool_for, close_pool, Pool
from ptb_api.storage_models import UploadInput
from ptb_api.quota import StorageError
from test_p07_policy_postgres import (legacy,migrate,
    test_unmigrated_database_read_delete_but_no_new_writes,
    test_last_byte_concurrent_reservations_and_database_growth_guard,
    test_upload_three_days_exact_expiry_and_download_no_renewal,
    test_delete_failure_keeps_used_and_reserved_then_retries,
    test_inflight_recovery_settles_without_growing_and_preserves_deadline,
    test_real_generated_and_zip_results_with_old_input_expiry,
    test_input_expiry_and_stale_worker_cannot_publish,
    test_inflight_completed_bytes_can_finalize_and_disk_full_does_not_reserve,
    test_over_quota_http_login_usage_download_and_delete_are_still_available)
from test_p07_publication_postgres import test_actual_publication_expiry_and_version
from web_postgres_support import cluster


@pytest.fixture
def env(legacy):
    migrate(legacy)
    yield legacy
    close_pool(legacy.dsn)


def other(e):
    owner,project=str(uuid4()),str(uuid4())
    e.conn.execute('INSERT INTO ptb_accounts.users(id,username,password_hash) VALUES(%s,%s,%s)',(owner,owner,'unused'))
    e.conn.execute('INSERT INTO ptb_accounts.projects(id,owner_id,name) VALUES(%s,%s,%s)',(project,owner,'fixture'))
    return owner,project


def upload(e,owner,project,data=b'content'):
    item=e.storage.create(owner,UploadInput(project_id=project,name='fixture.bin',idempotency_key=uuid4().hex,expected_bytes=len(data)))
    e.storage.append(owner,item['id'],0,data)
    return e.storage.finalize(owner,item['id'],hashlib.sha256(data).hexdigest())


def test_unrelated_account_progresses_while_first_holds_disk_scope(env):
    e=env;owner,project=other(e);item=upload(e,owner,project)
    with e.storage._locked(e.owner):
        with ThreadPoolExecutor(1) as threads:
            assert threads.submit(e.storage.read_block,owner,item['id'],0,7).result(timeout=2)==b'content'
    with pytest.raises(StorageError,match='asset_not_found'):
        e.storage.read_block(e.owner,item['id'],0,7)


def test_shared_readers_and_exclusive_maintenance(env):
    e=env;item=upload(e,e.owner,e.project);entered=threading.Event()
    def maintenance():
        with e.storage._locked():entered.set()
    with ThreadPoolExecutor(2) as threads:
        with e.storage._locked(e.owner,item['id'],read=True):
            assert threads.submit(e.storage.read_block,e.owner,item['id'],0,7).result(timeout=2)==b'content'
            future=threads.submit(maintenance)
            assert not entered.wait(.1)
        future.result(timeout=3)
    assert entered.is_set()


def test_pool_reuses_backend_and_resets_failed_transaction_and_locks(env):
    with connection(env.dsn) as c:
        pid=c.info.backend_pid
        c.execute('SELECT pg_advisory_lock(987654)')
        c.execute('SET default_transaction_read_only=on')
    with connection(env.dsn) as c:
        assert c.info.backend_pid==pid
        assert c.execute('SHOW default_transaction_read_only').fetchone()['default_transaction_read_only']=='off'
        assert not c.execute('SELECT 1 FROM pg_locks WHERE pid=pg_backend_pid() AND locktype=\'advisory\'').fetchone()
    assert pool_for(env.dsn).created==1


def test_pool_caps_connections_and_times_out_without_creating_more(env):
    pool=Pool(env.dsn,maximum=2,timeout=.15)
    try:
        with pool.connection(),pool.connection():
            with pytest.raises(Exception,match='Database pool busy'):
                with pool.connection():pass
        assert pool.created==2 and pool.peak==2
    finally:pool.close()


def test_shutdown_drains_active_borrower_before_allowing_replacement_pool(env):
    import psycopg
    with connection(env.dsn) as c:
        pid=c.info.backend_pid
        close_pool(env.dsn)
        with pytest.raises(psycopg.OperationalError,match='draining'):
            with connection(env.dsn):pass
        assert c.execute('SELECT 1 AS n').fetchone()['n']==1
    assert c.closed
    with connection(env.dsn) as replacement:
        assert replacement.info.backend_pid!=pid


def test_concurrent_uploads_and_repeated_downloads_reuse_bounded_connections(env,tmp_path):
    e=env;identities=[(e.owner,e.project)]+[other(e) for _ in range(7)]
    data=b'bounded-asset'*4096
    started=time.perf_counter()
    with ThreadPoolExecutor(8) as threads:
        assets=list(threads.map(lambda pair:upload(e,*pair,data=data),identities))
        def read(pair):
            (owner,_),asset=pair
            for _ in range(10):assert e.storage.read_block(owner,asset['id'],0,len(data))==data
        list(threads.map(read,zip(identities,assets)))
    pool=pool_for(e.dsn)
    assert pool.created<=8 and pool.peak<=8
    assert e.conn.execute('SELECT sum(used_bytes) AS n FROM ptb_storage.quota_accounts').fetchone()['n']==len(data)*8
    # The enclosing runner copies this small evidence out of disposable scratch.
    (tmp_path/'concurrency.json').write_text(json.dumps(dict(accounts=8,uploads=8,reads=80,
        bytes_per_asset=len(data),created_connections=pool.created,peak_connections=pool.peak,
        elapsed_seconds=time.perf_counter()-started,scope='Windows local synthetic PG; not production capacity')),encoding='utf-8')


def test_successful_login_resets_only_account_ip_bucket_with_real_constraint(env):
    from ptb_api.account_store import PostgresAccountStore,hash_token
    account=PostgresAccountStore(env.dsn)
    for _ in range(12):
        assert account.allow_login('synthetic','127.0.0.1')
        account.login_succeeded('synthetic','127.0.0.1')
    assert account.allow_login('synthetic','127.0.0.2')
    account.login_succeeded('synthetic','127.0.0.1')
    row=env.conn.execute('SELECT attempts FROM ptb_accounts.login_attempts WHERE bucket_key=%s',
        (hash_token('account-ip:synthetic\0'+'127.0.0.2'),)).fetchone()
    assert row['attempts']==1


def test_failed_transaction_does_not_leak_or_commit_into_next_borrower(env):
    import psycopg
    with pytest.raises(psycopg.errors.DivisionByZero):
        with connection(env.dsn) as c,c.transaction():
            c.execute("UPDATE ptb_accounts.projects SET name='rollback-probe' WHERE id=%s",(env.project,))
            c.execute('SELECT 1/0')
    with connection(env.dsn) as c:
        assert c.execute('SELECT name FROM ptb_accounts.projects WHERE id=%s',(env.project,)).fetchone()['name']=='synthetic'
        assert c.execute('SHOW lock_timeout').fetchone()['lock_timeout']=='5s'


def test_scoped_lock_allows_other_account_in_separate_process(env):
    import subprocess,sys
    e=env;owner,project=other(e);asset=upload(e,owner,project)
    code="import json,sys; from ptb_api.storage import Storage; d=json.load(sys.stdin); s=Storage(d['dsn'],d['root'],min_free_bytes=0); s.ready=True; assert s.read_block(d['owner'],d['asset'],0,7)==b'content'; print('ok')"
    with e.storage._locked(e.owner):
        result=subprocess.run([sys.executable,'-c',code],input=json.dumps(dict(dsn=e.dsn,root=str(e.root),owner=owner,asset=asset['id'])),
            text=True,capture_output=True,timeout=10,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='ok'


def test_download_rechecks_real_revoked_session_between_blocks(env):
    import asyncio
    from datetime import datetime,timedelta,timezone
    from ptb_api.account_store import PostgresAccountStore,hash_token
    from ptb_api.assets import PrivateDownload
    from ptb_api.quota import CHUNK_BYTES
    e=env;accounts=PostgresAccountStore(e.dsn);token=hash_token('public-test-token')
    user=accounts.get_user('synthetic')
    assert accounts.issue_session(user,token,'csrf',datetime.now(timezone.utc)+timedelta(hours=1),'')
    data=b'x'*CHUNK_BYTES
    asset=e.storage.create(e.owner,UploadInput(project_id=e.project,name='fixture.bin',expected_bytes=2*len(data),idempotency_key=uuid4().hex))
    for offset in (0,len(data)):e.storage.append(e.owner,asset['id'],offset,data)
    asset=e.storage.finalize(e.owner,asset['id']);blocks=[]
    class Context:
        def session(self,request):
            row=accounts.session(token)
            if not row:raise StorageError('session_expired',401)
            return dict(id=str(row['id']))
    async def send(message):
        if message['type']=='http.response.body' and message.get('body'):
            blocks.append(message['body'])
            # No storage lock/connection may remain held across the send.
            with e.storage._locked():pass
            accounts.revoke(token)
    response=PrivateDownload(e.storage,Context(),None,e.owner,asset,0,2*len(data),False)
    with pytest.raises(StorageError,match='session_expired'):asyncio.run(response({},None,send))
    assert blocks==[data]
