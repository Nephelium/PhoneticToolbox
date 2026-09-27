"""P06-REMOTE synthetic protocol tests. No science, lab host or production P07 claim.

Uses real independent SQLite transactions/connections and persistent test blobs.
This file's SyntheticFiles is deliberately NOT a production storage adapter.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import hashlib
import json
import sqlite3
import subprocess
import sys
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ptb_api.remote import create_remote_router
from ptb_api.remote_models import Output, Claim
from ptb_api.lpc_models import LpcTaskConfig
from ptb_worker.remote_scheduler import RemoteCoordinator, RemotePolicy
from ptb_worker.store import SQLiteJobStore, Transaction, JobError, canonical

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = 'a'*64
PAYLOAD = b'public synthetic bytes'
PROJECT = str(uuid4())


class SyntheticFiles:
    """Persistent synthetic ledger to exercise transaction rollback/idempotency."""
    def validate_inputs(self, tx, job):
        rows = tx.execute('SELECT * FROM test_inputs WHERE job_id=?', (job['id'],)).fetchall()
        if any(r['expires_at'] <= tx.now() or not r['active'] for r in rows):
            raise JobError('input_unavailable', 410)
        return [dict(id=r['id'], sha256=hashlib.sha256(r['data']).hexdigest(), size_bytes=len(r['data']), expires_at=r['expires_at']) for r in rows]

    def read(self, tx, job, asset_id, offset, size):
        return tx.execute('SELECT data FROM test_inputs WHERE id=? AND job_id=?', (asset_id, job['id'])).fetchone()['data'][offset:offset+size]

    def reserve(self, tx, job, uid, name, size):
        quota = tx.execute('SELECT * FROM test_quota WHERE owner=?', (job['owner_id'],)).fetchone()
        if quota['used']+quota['reserved']+size > quota['budget']: raise JobError('quota_exceeded', 413)
        tx.execute('UPDATE test_quota SET reserved=reserved+? WHERE owner=?', (size, job['owner_id']))
        tx.execute('INSERT INTO test_blobs(id,owner,data,size) VALUES(?,?,?,?)', (uid, job['owner_id'], b'', size))

    def append(self, tx, job, uid, offset, data):
        row = tx.execute('SELECT * FROM test_blobs WHERE id=?', (uid,)).fetchone()
        assert len(row['data']) == offset
        tx.execute('UPDATE test_blobs SET data=? WHERE id=?', (row['data']+data, uid))

    def seal(self, tx, job, uid, expected_hash):
        data = tx.execute('SELECT data FROM test_blobs WHERE id=?', (uid,)).fetchone()['data']
        if hashlib.sha256(data).hexdigest() != expected_hash: raise JobError('hash_mismatch', 422)

    def publish(self, tx, job, uploads):
        size = sum(u['size_bytes'] for u in uploads)
        tx.execute('UPDATE test_quota SET used=used+?,reserved=reserved-?,settlements=settlements+1 WHERE owner=?', (size, size, job['owner_id']))
        return {'files': [dict(id=u['id'], sha256=u['sha256']) for u in uploads]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    path = tmp_path/'remote.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript((ROOT/'backend/migrations/002_jobs_sqlite.sql').read_text('utf-8'))
        # Same review SQL, only namespace removed for independent SQLite evidence.
        db.executescript((ROOT/'backend/migrations/007_remote.sql').read_text('utf-8').replace('ptb_jobs.', ''))
        db.executescript('''CREATE TABLE test_inputs(id TEXT PRIMARY KEY,job_id TEXT,data BLOB,expires_at REAL,active INTEGER DEFAULT 1);
            CREATE TABLE test_quota(owner TEXT PRIMARY KEY,budget INTEGER DEFAULT 1000000000,used INTEGER DEFAULT 0,reserved INTEGER DEFAULT 0,settlements INTEGER DEFAULT 0);
            CREATE TABLE test_blobs(id TEXT PRIMARY KEY,owner TEXT,data BLOB,size INTEGER);''')
    clock = [1000.0]
    monkeypatch.setattr(Transaction, 'now', lambda self: clock[0])
    store = SQLiteJobStore(path, max_running=1)
    files = SyntheticFiles()
    coordinator = RemoteCoordinator(store, files, server_capability=lambda op, runtime: RUNTIME)
    e = SimpleNamespace(store=store, files=files, c=coordinator, clock=clock, path=path)

    def node(owner='owner', healthy=True):
        ident, token = coordinator.register(scopes=[[owner, PROJECT]], operations=['lpc_analysis'], runtime_hash=RUNTIME, slots=1, memory_bytes=2_000_000_000)
        if healthy:
            for _ in range(4):
                coordinator.health(token, RUNTIME); clock[0] += 10
        return ident, token

    def job(owner='owner', memory=100_000_000, deadline=300, attempts=3):
        jid, asset = str(uuid4()), str(uuid4())
        with store.transaction() as tx:
            tx.execute('INSERT INTO test_quota(owner) VALUES(?) ON CONFLICT(owner) DO NOTHING', (owner,))
            tx.execute('''INSERT INTO jobs(id,owner_id,project_id,idempotency_key,request_hash,snapshot,state,deadline,created_at,updated_at)
                VALUES(?,?,?,?,?,?,'queued',?,?,?)''', (jid, owner, PROJECT, str(uuid4()), 'b'*64,
                canonical(dict(operation='lpc_analysis', config=dict(inputs=[asset],
                    analysis=LpcTaskConfig(roi_end=0.1).model_dump(), max_output_bytes=1000),
                    input_refs={}, core_version='synthetic', adapter_version='synthetic')),
                clock[0]+deadline, clock[0], clock[0]))
            tx.execute('INSERT INTO test_inputs(id,job_id,data,expires_at) VALUES(?,?,?,?)', (asset, jid, PAYLOAD, clock[0]+1000))
        coordinator.enroll_job(jid, runtime_hash=RUNTIME, memory_bytes=memory, max_output_bytes=1000, max_attempts=attempts)
        return jid

    e.node, e.job = node, job
    return e


def claim(e, token):
    return e.c.claim(token, str(uuid4()), RUNTIME)


def downloaded(e, token, a):
    for i in a['inputs']:
        assert e.c.read(token, a['attempt_id'], a['generation'], i['id'], 0, 100) == PAYLOAD
    e.c.heartbeat(token, a['attempt_id'], a['generation'], 'compute')


def uploaded(e, token, a):
    downloaded(e, token, a)
    e.c.heartbeat(token, a['attempt_id'], a['generation'], 'upload')
    body = Output(generation=a['generation'], key='result', name='result.json', size_bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())
    u = e.c.output(token, a['attempt_id'], body)
    e.c.write(token, a['attempt_id'], a['generation'], u['upload_id'], 0, b'abc', body.sha256)
    return body, u


def row(e, table, where='', args=()):
    with e.store.transaction(write=False) as tx:
        return dict(tx.execute('SELECT * FROM '+table+' '+where, args).fetchone())


def test_two_nodes_compete_and_lost_claim_response(env):
    e = env; _, t1 = e.node(); _, t2 = e.node(); e.c.health(t1, RUNTIME); e.job()
    barrier = threading.Barrier(2)
    def compete(t):
        barrier.wait(); return claim(e, t)
    with ThreadPoolExecutor(2) as pool: values = list(pool.map(compete, [t1, t2]))
    assert sum(x is not None for x in values) == 1
    a = next(x for x in values if x); Claim.model_validate(a)
    attempt = row(e, 'remote_attempts'); token = t1 if values[0] else t2
    assert e.c.claim(token, attempt['request_id'], RUNTIME) == a
    assert row(e, 'jobs')['generation'] == 1


def test_node_server_compete_and_node_preference(env):
    e = env; _, t = e.node(); e.job()
    with ThreadPoolExecutor(2) as pool:
        n = pool.submit(claim, e, t); s = pool.submit(e.c.claim_server)
        assert n.result() is not None; assert s.result() is None
    assert row(e, 'jobs')['generation'] == 1


def test_claim_grace_is_bounded_even_healthy_idle_node(env):
    e = env; _, t = e.node(); e.job(); assert e.c.claim_server() is None
    for _ in range(3): e.clock[0] += 10; e.c.health(t, RUNTIME)
    assert e.c.claim_server()['location'] == 'server'


def test_server_one_slot_account_limit_and_remote_parallel(env):
    e = env; _, t = e.node(); e.job('other'); server = e.c.claim_server(); assert server
    e.job(); assert claim(e, t)
    e.job('third'); assert e.c.claim_server() is None
    e.job(); assert claim(e, t) is None
    assert e.store.max_running == 1 and e.store.lease_seconds == 10


def test_download_retry_progress_and_stall(env):
    e = env; _, t = e.node(); e.job(); a = claim(e, t); i = a['inputs'][0]['id']
    assert e.c.read(t, a['attempt_id'], 1, i, 0, 5) == PAYLOAD[:5]
    e.clock[0] += 20
    assert e.c.read(t, a['attempt_id'], 1, i, 0, 5) == PAYLOAD[:5]
    e.c.heartbeat(t, a['attempt_id'], 1, 'download'); e.clock[0] += 11
    e.c.recover(); assert row(e, 'remote_attempts')['error_code'] == 'transfer_stalled'
    assert e.c.claim_server()['generation'] == 2


def test_compute_network_loss_waits_for_lease_then_fences_late_node(env):
    e = env; _, t = e.node(); e.job(); a = claim(e, t); downloaded(e, t, a)
    e.clock[0] += 31; assert e.c.claim_server() is None
    e.clock[0] += 30; b = e.c.claim_server(); assert b['generation'] == 2
    with pytest.raises(JobError, match='stale_attempt'): e.c.read(t, a['attempt_id'], 1, a['inputs'][0]['id'], 0, 1)


def test_heartbeat_does_not_hide_upload_stall(env):
    e = env; _, t = e.node(); e.job(); a = claim(e, t); downloaded(e, t, a)
    e.c.heartbeat(t, a['attempt_id'], 1, 'upload')
    for _ in range(2): e.clock[0] += 10; e.c.heartbeat(t, a['attempt_id'], 1, 'upload')
    e.clock[0] += 11
    with pytest.raises(JobError, match='transfer_stalled'): e.c.heartbeat(t, a['attempt_id'], 1, 'upload')
    e.c.recover(); assert row(e, 'jobs')['state'] == 'queued'


def test_complete_lost_response_duplicate_blocks_and_service_restart(env):
    e = env; _, t = e.node(); e.job(); a = claim(e, t); body, u = uploaded(e, t, a)
    assert e.c.output(t, a['attempt_id'], body)['upload_id'] == u['upload_id']
    assert e.c.write(t, a['attempt_id'], 1, u['upload_id'], 0, b'abc', body.sha256) == {'offset': 3}
    result = e.c.complete(t, a['attempt_id'], 1)
    e.c = RemoteCoordinator(SQLiteJobStore(e.path), SyntheticFiles())
    e.clock[0] += 400  # receipt retry after original execution deadline, token still valid
    assert e.c.complete(t, a['attempt_id'], 1) == result
    assert row(e, 'test_quota')['settlements'] == 1
    assert row(e, 'test_quota')['used'] == 3 and row(e, 'test_quota')['reserved'] == 0
    with pytest.raises(JobError): e.c.write(t, a['attempt_id'], 1, u['upload_id'], 0, b'abc', body.sha256)


def test_cancel_complete_race_serializes(env):
    e = env; _, t = e.node(); jid = e.job(); a = claim(e, t); uploaded(e, t, a)
    barrier = threading.Barrier(2)
    def complete():
        barrier.wait()
        try: return e.c.complete(t, a['attempt_id'], 1)
        except JobError as error: assert error.code == 'cancelled'
    def cancel():
        barrier.wait(); return e.store.cancel('owner', jid)
    with ThreadPoolExecutor(2) as pool:
        one, two = pool.submit(complete), pool.submit(cancel); one.result(); two.result()
    e.c.recover(); j = row(e, 'jobs'); q = row(e, 'test_quota')
    assert (j['state'], q['settlements']) in [('succeeded', 1), ('cancelled', 0)]


@pytest.mark.parametrize('fault', ['revoke', 'expire', 'scope', 'token'])
def test_authority_loss_rejects_every_attempt_operation(env, fault):
    e = env; node, t = e.node(); e.job(); a = claim(e, t); body, u = uploaded(e, t, a)
    if fault == 'revoke': e.c.revoke(node)
    else:
        with e.store.transaction() as tx:
            if fault == 'expire': tx.execute('UPDATE test_inputs SET expires_at=?', (e.clock[0],))
            elif fault == 'scope': tx.execute("UPDATE remote_nodes SET scopes='[]'")
            else: tx.execute('UPDATE remote_nodes SET token_until=?', (e.clock[0],))
    actions = [lambda: e.c.read(t, a['attempt_id'], 1, a['inputs'][0]['id'], 0, 1),
        lambda: e.c.heartbeat(t, a['attempt_id'], 1, 'upload'), lambda: e.c.output(t, a['attempt_id'], body),
        lambda: e.c.write(t, a['attempt_id'], 1, u['upload_id'], 0, b'abc', body.sha256),
        lambda: e.c.complete(t, a['attempt_id'], 1), lambda: e.c.fail(t, a['attempt_id'], 1, 'network_error')]
    for action in actions:
        with pytest.raises(JobError): action()
    assert row(e, 'test_quota')['settlements'] == 0


def test_failed_attempt_reservation_stays_charged_until_physical_cleanup(env):
    e = env; _, t = e.node(); e.job(); a = claim(e, t); uploaded(e, t, a)
    e.c.fail(t, a['attempt_id'], 1, 'network_error')
    assert row(e, 'test_quota')['reserved'] == 3
    with e.store.transaction() as tx: tx.execute('UPDATE test_quota SET budget=3')
    # Rehabilitate node; old bytes still count, new reservation cannot hide them.
    for _ in range(4): e.clock[0] += 10; e.c.health(t, RUNTIME)
    b = claim(e, t); downloaded(e, t, b); e.c.heartbeat(t, b['attempt_id'], 2, 'upload')
    with pytest.raises(JobError, match='quota_exceeded'):
        e.c.output(t, b['attempt_id'], Output(generation=2, key='x', name='x', size_bytes=1, sha256='a'*64))


@pytest.mark.parametrize('code', ['algorithm_error', 'invalid_input', 'missing_model', 'authentication_failed', 'cancelled', 'resource_limit'])
def test_non_network_failures_do_not_retry(env, code):
    e = env; _, t = e.node(); e.job(); a = claim(e, t)
    e.c.fail(t, a['attempt_id'], 1, code)
    assert e.c.claim_server() is None
    assert row(e, 'jobs')['state'] in ('failed', 'cancelled')


def test_retry_count_and_total_deadline(env):
    e = env; _, t = e.node(); e.job(attempts=2); a = claim(e, t)
    e.c.fail(t, a['attempt_id'], 1, 'network_error'); b = e.c.claim_server()
    e.clock[0] += 11; e.c.recover()
    assert row(e, 'jobs')['state'] == 'failed' and e.c.claim_server() is None
    e.job('next', deadline=1); e.clock[0] += 2; e.c.recover()
    assert row(e, 'jobs', 'WHERE owner_id=?', ('next',))['error_code'] == 'deadline_exceeded'


def test_flapping_requires_stable_recovery_and_does_not_migrate_server(env):
    e = env; _, t = e.node(); e.job(); a = claim(e, t); downloaded(e, t, a)
    e.clock[0] += 61; b = e.c.claim_server(); assert b['location'] == 'server'
    e.job('next'); node2, t2 = e.node('next', healthy=False)
    worker = row(e, 'remote_attempts', 'WHERE id=?', (b['attempt_id'],))['worker_id']
    def server_advance(seconds):
        for _ in range(seconds):
            e.clock[0] += 1
            assert e.store.heartbeat(b['job_id'], worker, b['generation'], 0.1) == 'running'
    for _ in range(3):
        server_advance(31); e.c.health(t2, RUNTIME)
        assert claim(e, t2) is None
    for _ in range(3): server_advance(10); e.c.health(t2, RUNTIME)
    c = claim(e, t2); assert c['location'] == node2
    assert row(e, 'jobs', 'WHERE id=?', (b['job_id'],))['generation'] == b['generation']
    assert row(e, 'jobs', 'WHERE id=?', (b['job_id'],))['state'] == 'running'


def test_server_budget_and_capability_fail_closed(env):
    e = env; e.job(memory=2_000_000_000)
    assert e.c.claim_server() is None and row(e, 'remote_jobs')['reason'] == 'server_over_budget'
    e.c.server_capability = lambda op, runtime: False
    assert e.c.claim_server() is None and row(e, 'remote_jobs')['reason'] == 'server_capability_unavailable'


def test_runtime_and_scope_cannot_self_authorize(env):
    e = env; _, t = e.node('other'); e.job()
    assert claim(e, t) is None
    with pytest.raises(JobError, match='runtime_mismatch'): e.c.claim(t, str(uuid4()), 'b'*64)


def test_publish_rollback_is_atomic(env, monkeypatch):
    e = env; _, t = e.node(); e.job(); a = claim(e, t); uploaded(e, t, a)
    old = e.files.publish
    def broken(*args): old(*args); raise JobError('injected_publish_failure')
    monkeypatch.setattr(e.files, 'publish', broken)
    with pytest.raises(JobError): e.c.complete(t, a['attempt_id'], 1)
    assert row(e, 'test_quota')['settlements'] == 0
    assert row(e, 'jobs')['result_manifest'] is None
    monkeypatch.setattr(e.files, 'publish', old)
    assert e.c.complete(t, a['attempt_id'], 1)


def test_http_boundaries_and_schema(env):
    e = env; _, t = e.node(); e.job()
    app = FastAPI(); app.include_router(create_remote_router(e.c))
    headers = {'Authorization': 'Bearer '+t}
    with TestClient(app, base_url='https://synthetic.test') as c:
        body = dict(request_id=str(uuid4()), runtime_hash=RUNTIME)
        assert c.post('/api/v1/worker/poll', json=body).status_code == 401
        assert c.post('/api/v1/worker/poll', content=b'x'*16385, headers=headers).status_code == 413
        assert c.post('/api/v1/worker/poll', json=body|{'shell': 'bad'}, headers=headers).status_code == 422
        a = c.post('/api/v1/worker/poll', json=body, headers=headers).json()
        assert c.get(f"/api/v1/worker/attempts/{a['attempt_id']}/inputs/{uuid4()}?generation=1", headers=headers).status_code == 404
        assert c.put(f"/api/v1/worker/attempts/{a['attempt_id']}/outputs/{uuid4()}?generation=1&offset=0", content=b'x'*1_048_577, headers=headers).status_code == 413
    with TestClient(app, base_url='http://synthetic.test') as c:
        assert c.post('/api/v1/worker/poll', json=body, headers=headers).status_code == 401


def test_claim_retry_returns_remaining_lease_not_initial_duration(env):
    e = env; _, t = e.node(); e.job(); key = str(uuid4())
    a = e.c.claim(t, key, RUNTIME); e.clock[0] += 12
    b = e.c.claim(t, key, RUNTIME)
    assert b['attempt_id'] == a['attempt_id'] and b['lease_until'] == a['lease_until']
    assert b['lease_remaining_seconds'] == 48
    assert b['deadline_remaining_seconds'] == a['deadline_remaining_seconds']-12
    assert b['inputs'][0]['expires_remaining_seconds'] == a['inputs'][0]['expires_remaining_seconds']-12


def test_cancel_wins_prevents_publish(env):
    e = env; _, t = e.node(); jid = e.job(); a = claim(e, t); uploaded(e, t, a)
    e.store.cancel('owner', jid)
    with pytest.raises(JobError, match='cancelled'): e.c.complete(t, a['attempt_id'], 1)
    e.c.recover(); assert row(e, 'jobs')['state'] == 'cancelled'
    assert row(e, 'test_quota')['settlements'] == 0


def test_old_generation_cannot_complete_new_attempt(env):
    e = env; _, t = e.node(); e.job(); a = claim(e, t); uploaded(e, t, a)
    e.c.fail(t, a['attempt_id'], 1, 'network_error'); assert e.c.claim_server()['generation'] == 2
    with pytest.raises(JobError, match='stale_attempt'): e.c.complete(t, a['attempt_id'], 1)
    assert row(e, 'test_quota')['settlements'] == 0


def test_invalid_hash_and_conflicting_block_do_not_publish(env):
    e = env; _, t = e.node(); e.job(); a = claim(e, t); body, u = uploaded(e, t, a)
    with pytest.raises(JobError, match='hash_mismatch'):
        e.c.write(t, a['attempt_id'], 1, u['upload_id'], 0, b'abc', 'a'*64)
    with pytest.raises(JobError, match='idempotency_conflict'):
        e.c.write(t, a['attempt_id'], 1, u['upload_id'], 0, b'def', hashlib.sha256(b'def').hexdigest())
    assert row(e, 'test_quota')['reserved'] == 3 and row(e, 'test_quota')['used'] == 0


def test_output_and_total_admission_budgets(env):
    e = env; _, t = e.node(); e.job(); a = claim(e, t); downloaded(e, t, a)
    e.c.heartbeat(t, a['attempt_id'], 1, 'upload')
    with pytest.raises(JobError, match='output_budget_exceeded'):
        e.c.output(t, a['attempt_id'], Output(generation=1, key='x', name='x', size_bytes=1001, sha256='a'*64))
    e.c.policy = RemotePolicy(total_active=1)
    with pytest.raises(JobError, match='remote_queue_full'): e.job('another')


def test_node_memory_is_aggregate_across_slots(env):
    e = env; ident, t = e.node(); e.job(memory=1_200_000_000); assert claim(e, t)
    with e.store.transaction() as tx:
        tx.execute('UPDATE remote_nodes SET slots=2,scopes=? WHERE id=?', (canonical([['owner', PROJECT], ['other', PROJECT]]), ident))
    e.job('other', memory=1_200_000_000)
    assert claim(e, t) is None


def test_fast_health_requests_do_not_bypass_stabilization(env):
    e = env; _, t = e.node(healthy=False); e.job()
    for _ in range(10): e.c.health(t, RUNTIME)
    assert claim(e, t) is None
    assert row(e, 'remote_nodes')['healthy_count'] == 1


def test_credential_rotation_invalidates_old_token(env):
    e = env; ident, t = e.node(); e.job(); a = claim(e, t)
    new = e.c.issue(ident)
    with pytest.raises(JobError, match='node_unauthorized'): e.c.check(t, a['attempt_id'], 1)
    e.c.check(new, a['attempt_id'], 1)


def test_lifetime_expiring_during_publish_rolls_back_ledger(env, monkeypatch):
    e = env; _, t = e.node(); e.job(); a = claim(e, t); uploaded(e, t, a)
    old = e.files.publish
    def expired(*args):
        result = old(*args); e.clock[0] += 1000; return result
    monkeypatch.setattr(e.files, 'publish', expired)
    with pytest.raises(JobError): e.c.complete(t, a['attempt_id'], 1)
    assert row(e, 'test_quota')['settlements'] == 0 and row(e, 'jobs')['result_manifest'] is None


def test_expired_queued_input_and_unregistered_operation(env):
    e = env; jid = e.job()
    with e.store.transaction() as tx: tx.execute('UPDATE test_inputs SET active=0')
    assert e.c.claim_server() is None
    assert row(e, 'jobs')['error_code'] == 'input_unavailable'
    with pytest.raises(ValueError):
        e.c.register(scopes=[['owner', PROJECT]], operations=['shell'], runtime_hash=RUNTIME, slots=1, memory_bytes=1)


def test_normal_node_exit_explicitly_releases_lease_without_user_cancel(env):
    e = env; _, t = e.node(); e.job(); a = claim(e, t)
    e.c.fail(t, a['attempt_id'], 1, 'node_shutdown')
    assert row(e, 'jobs')['state'] == 'queued'
    assert e.c.claim_server()['generation'] == 2
    assert row(e, 'remote_attempts', 'WHERE id=?', (a['attempt_id'],))['error_code'] == 'node_shutdown'


def test_server_runtime_records_actual_verified_fingerprint(env):
    e = env; e.job(); e.c.server_capability = lambda op, required: 'b'*64
    a = e.c.claim_server()
    assert a['runtime_hash'] == 'b'*64
    assert row(e, 'remote_jobs')['runtime_hash'] == RUNTIME


@pytest.mark.parametrize('boundary', ['input', 'credential'])
def test_lease_never_outlives_input_or_credential(env, boundary):
    e = env; _, t = e.node(); e.job()
    with e.store.transaction() as tx:
        if boundary == 'input': tx.execute('UPDATE test_inputs SET expires_at=?', (e.clock[0]+15,))
        else: tx.execute('UPDATE remote_nodes SET token_until=?', (e.clock[0]+15,))
    a = claim(e, t); assert a['lease_remaining_seconds'] == 15
    e.clock[0] += 5
    assert e.c.heartbeat(t, a['attempt_id'], 1, 'download')['lease_remaining_seconds'] == 10


def test_complete_receipt_survives_fresh_python_process(env):
    e = env; _, t = e.node(); e.job(); a = claim(e, t); uploaded(e, t, a)
    result = e.c.complete(t, a['attempt_id'], 1)
    # Ephemeral test token goes through stdin, never argv, environment or stdout.
    script = '''import json,sys
from ptb_worker.store import SQLiteJobStore,Transaction
from ptb_worker.remote_scheduler import RemoteCoordinator
p=json.load(sys.stdin)
Transaction.now=lambda self:p['now']
class NoFileAccess:
 def __getattr__(self,name):raise AssertionError('receipt must not touch files')
c=RemoteCoordinator(SQLiteJobStore(p['path']),NoFileAccess())
print(json.dumps(c.complete(p['token'],p['attempt'],1)))
'''
    child = subprocess.run([sys.executable, '-c', script], input=json.dumps(dict(path=str(e.path),
        now=e.clock[0], token=t, attempt=a['attempt_id'])), text=True, capture_output=True, timeout=15)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == result
    assert row(e, 'test_quota')['settlements'] == 1
