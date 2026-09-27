"""Real PG publication/transition tests on the fresh, opt-in synthetic cluster.

Payloads are arbitrary bytes, not scientific module verification.
"""
import hashlib
import json
import time
from uuid import uuid4
from unittest.mock import patch

import pytest
from ptb_api.job_models import JobView
from ptb_api.quota import StorageError
from ptb_worker.acoustic_files import AcousticFiles
from ptb_worker.store import Transaction, JobError
from test_p07_policy_postgres import cluster, legacy, migrate, old_asset, pipeline
from test_p07_policy_wiring import CASES
from pathlib import Path


@pytest.mark.parametrize('operation,model,names', CASES)
def test_actual_publication_expiry_and_version(legacy, operation, model, names):
    e = legacy
    source, source_deadline = old_asset(e, expires=time.time()+3600)
    migrate(e)
    jobs, _, submit, claim = pipeline(e)
    files = AcousticFiles(jobs, e.storage)
    job = submit(inputs=[source])
    snapshot = json.loads(e.conn.execute('SELECT snapshot FROM ptb_jobs.jobs WHERE id=%s', (job['id'],)).fetchone()['snapshot'])
    snapshot['operation'] = operation
    e.conn.execute('UPDATE ptb_jobs.jobs SET snapshot=%s WHERE id=%s', (json.dumps(snapshot), job['id']))
    _, identity = claim(job)
    for name in names:
        out = files.output(identity, name, 'result', 1)
        files.write(identity, out['id'], 0, b'x')
        files.seal(identity, out['id'])
    published = time.time()
    with patch.object(Transaction, 'now', return_value=published):
        manifest = files.complete(identity)
    expected = source_deadline if operation in ('archive_zip', 'extract_zip', 'textgrid_segment') else published+259200
    assert manifest['policy_version'] == 2
    assert all(f['expires_at'] == expected for f in manifest['files'])
    assert JobView.model_validate(jobs.get(e.owner, job['id'])).result_manifest.policy_version == 2
    for f in manifest['files']:
        assert e.storage.metadata(e.owner, f['id'])['policy_version'] == 2
        assert e.storage.read_block(e.owner, f['id'], 0, 1) == b'x'
        assert e.storage.metadata(e.owner, f['id'])['expires_at'] == expected
    assert e.storage.metadata(e.owner, source)['expires_at'] == source_deadline


@pytest.mark.parametrize('copy_fields', [
    {},
    {'saved_copy': True}, {'source_ref': {'sha256': 'b'*64}}, {'copy_result': {}},
    {'saved_copy': True, 'source_ref': {'sha256': 'b'*64}, 'copy_result': {}},
])
def test_old_inflight_m08_publication_version_matches_semantics(legacy, copy_fields):
    e = legacy
    source, deadline = old_asset(e, b'copied-pcm', expires=time.time()+3600)
    output, _ = old_asset(e, b'copied-pcm', state='uploading')
    now = time.time()
    job_id = str(uuid4())
    snapshot = dict(operation='pitch_manipulation', config={'inputs': [source], 'max_output_bytes': 5_000_000_000}, **copy_fields)
    e.conn.execute('''INSERT INTO ptb_jobs.jobs(id,owner_id,project_id,idempotency_key,request_hash,snapshot,
        state,deadline,created_at,updated_at,worker_id,generation,lease_until)
        VALUES(%s,%s,%s,%s,%s,%s,'running',%s,%s,%s,'old-worker',1,%s)''',
        (job_id, e.owner, e.project, uuid4().hex, 'a'*64, json.dumps(snapshot), now+300, now, now, now+60))
    e.conn.execute("UPDATE ptb_storage.assets SET kind='result',name='saved.wav',sha256=%s WHERE id=%s",
                   (hashlib.sha256(b'copied-pcm').hexdigest(), output))
    for asset_id, role, generation in ((source, 'input', 0), (output, 'output', 1)):
        e.conn.execute('''INSERT INTO ptb_storage.job_assets(job_id,asset_id,owner_id,project_id,role,
            generation,input_sha256,input_expires_at,created_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
            (job_id, asset_id, e.owner, e.project, role, generation,
             hashlib.sha256(b'copied-pcm').hexdigest() if role=='input' else None,
             deadline if role=='input' else None, now))
    # Grandfathered reservation makes the account over quota at migration.
    old_asset(e, b'', reserved=1_000_000_001, state='uploading')
    migrate(e)
    assert e.storage.metadata(e.owner, source)['policy_version'] == 1
    assert e.conn.execute('SELECT policy_version FROM ptb_storage.assets WHERE id=%s', (output,)).fetchone()['policy_version'] == 1
    jobs, _, _, _ = pipeline(e)
    files = AcousticFiles(jobs, e.storage)
    identity = (job_id, 'old-worker', 1)
    with pytest.raises(StorageError, match='stale_worker'):
        files.complete((job_id, 'old-worker', 2))
    balance = e.storage.usage(e.owner)
    published = time.time()
    with patch.object(Transaction, 'now', return_value=published):
        manifest = files.complete(identity)
    expected = deadline if copy_fields else published+259200
    assert manifest['policy_version'] == 2 and manifest['files'][0]['expires_at'] == expected
    assert e.storage.metadata(e.owner, output)['policy_version'] == 2
    assert e.storage.read_block(e.owner, output, 0, 100) == b'copied-pcm'
    assert e.storage.usage(e.owner) == balance
    assert e.storage.metadata(e.owner, source)['expires_at'] == deadline


def test_retry_of_derived_job_keeps_original_source_deadline(legacy):
    e = legacy
    source, deadline = old_asset(e, expires=time.time()+3600)
    migrate(e)
    jobs, files, submit, claim = pipeline(e)
    first = submit('archive_zip', [source])
    _, identity = claim(first)
    files.fail(identity, 'execution_failed')
    retry = jobs.retry(e.owner, first['id'], uuid4().hex)
    _, identity = claim(retry)
    out = files.output(identity, 'copy.zip', 'archive', 1)
    files.write(identity, out['id'], 0, b'x'); files.seal(identity, out['id'])
    assert files.complete(identity)['files'][0]['expires_at'] == deadline
    assert e.storage.metadata(e.owner, source)['expires_at'] == deadline


def test_legacy_oversized_retry_is_a_public_error_not_validation_crash(legacy):
    e = legacy
    migrate(e)
    jobs, files, submit, claim = pipeline(e)
    first = submit()
    _, identity = claim(first)
    files.fail(identity, 'execution_failed')
    old = e.conn.execute('SELECT snapshot FROM ptb_jobs.jobs WHERE id=%s', (first['id'],)).fetchone()['snapshot']
    data = json.loads(old); data['config']['max_output_bytes'] = 5_000_000_000
    old = json.dumps(data)
    e.conn.execute('UPDATE ptb_jobs.jobs SET snapshot=%s WHERE id=%s', (old, first['id']))
    with pytest.raises(JobError, match='output_budget_exceeded'):
        jobs.retry(e.owner, first['id'], uuid4().hex)
    assert e.conn.execute('SELECT snapshot FROM ptb_jobs.jobs WHERE id=%s', (first['id'],)).fetchone()['snapshot'] == old


def test_reviewed_runner_is_target_bound_and_preserves_metadata(legacy, monkeypatch):
    e = legacy
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    from p07_policy_database import apply_checked, reviewed_sql, REVIEWED_SHA256
    old_asset(e, b'old-result')
    old_asset(e, b'', reserved=1_000_000_001, state='uploading')
    with pytest.raises(ValueError, match='reviewed_sql_hash_mismatch'):
        reviewed_sql('0'*64)
    sql = reviewed_sql(REVIEWED_SHA256)
    database = e.conn.execute('SELECT current_database() AS name').fetchone()['name']
    instance = str(e.conn.execute('SELECT instance_id FROM ptb_storage.state').fetchone()['instance_id'])
    with e.storage._locked() as conn:
        with pytest.raises(ValueError, match='target_identity_mismatch'):
            apply_checked(conn, e.root, database+'wrong', instance, sql)
        assert 'policy_version' not in conn.execute('SELECT * FROM ptb_storage.state').fetchone()
        result = apply_checked(conn, e.root, database, instance, sql)
    assert result['policy_version'] == 2 and not result['writes_reopened']
    assert e.storage.usage(e.owner)['over_quota']
