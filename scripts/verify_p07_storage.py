"""Prepared P07 SQL/filesystem scenarios. Never run without the named test-file approval."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
from unittest.mock import patch
from uuid import uuid4
from fastapi.testclient import TestClient
from ptb_api.account_store import PostgresAccountStore
from ptb_api.auth import AuthSettings
from ptb_api.main import create_app
from ptb_api.quota import CHUNK_BYTES, StorageError
from ptb_api.storage import Storage
from ptb_api.storage_models import UploadInput

ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT/'output/validation/p07/storage'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-p07-schema-and-test-files', action='store_true')
    if not parser.parse_args().approved_p07_schema_and_test_files:
        parser.error('Requires explicit P07 schema/generated-file cleanup approval')
    config = json.loads(sys.stdin.readline())
    store = Storage(config['dsn'], FILES, min_free_bytes=0)
    assert store.root == FILES and FILES.is_dir()
    accounts = PostgresAccountStore(config['dsn'])
    with accounts.connection() as conn:
        assert conn.execute('SELECT current_database() AS name').fetchone()['name'] == 'ptb_p05_test_20260909'
    store.recover()
    suffix = uuid4().hex[:10]
    people, created, checks = [], [], {}
    for letter in ('a', 'b'):
        name, password = f'p07_{letter}_{suffix}', secrets.token_urlsafe(32)
        user = accounts.create_user(name, password)
        project = accounts.create_project(str(user['id']), 'P07 合成文件验证')
        people.append((str(user['id']), str(project['id']), name, password))
    owner, project, *_ = people[0]

    def expect(code, action):
        try:
            action()
        except StorageError as error:
            assert error.code == code, error.code
        else:
            raise AssertionError('Expected ' + code)

    def create(expected=None, key=None):
        body = UploadInput(project_id=project, name='合成数据 ɑ.bin', expected_bytes=expected,
                           idempotency_key=key or uuid4().hex)
        asset = store.create(owner, body)
        created.append(asset['id'])
        return asset, body

    def upload(data):
        asset, _ = create(len(data))
        for offset in range(0, len(data), CHUNK_BYTES):
            store.append(owner, asset['id'], offset, data[offset:offset+CHUNK_BYTES])
        return store.finalize(owner, asset['id'], hashlib.sha256(data).hexdigest())

    try:
        # Q11 concurrent idempotency is real PG and an actual shared disk lock.
        asset, body = create(100)
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(lambda _: store.create(owner, body)['id'], range(4)))
        assert set(ids) == {asset['id']}
        expect('idempotency_conflict', lambda: store.create(owner, body.model_copy(update={'name': 'different'})))
        before = store.usage(owner)
        result = store.append(owner, asset['id'], 0, b'a'*30)
        after = store.usage(owner)
        assert after['used_bytes'] == before['used_bytes']+30
        assert after['reserved_bytes'] == before['reserved_bytes']-30
        assert result['reserved_bytes'] == 70
        assert store.append(owner, asset['id'], 0, b'a'*30) == result
        expect('chunk_conflict', lambda: store.append(owner, asset['id'], 0, b'b'*30))
        checks['Q01_transfer_no_double_count_Q11_idempotency'] = True

        asset, _ = create()
        store.append(owner, asset['id'], 0, b'unknown-length')
        assert store.finalize(owner, asset['id'])['size_bytes'] == 14
        assert store.usage(owner)['used_bytes'] >= 44
        checks['Q02_unknown_length'] = True

        # Q08/Q10 simulate process loss immediately after fsync, before settlement.
        # Only this script's reserved test file is mutated; not a real power cut.
        asset, _ = create(50)
        with store._locked() as conn:
            path = store._path(asset['id'])
            assert path.resolve().is_relative_to(FILES.resolve())
            with path.open('ab') as target:
                target.write(b'interrupted-write')
                target.flush(); os.fsync(target.fileno())
        reopened = Storage(config['dsn'], FILES, min_free_bytes=0)
        reopened.recover()
        state = next(a for a in reopened.list(owner, project) if a['id'] == asset['id'])
        assert state['size_bytes'] == 17 and state['reserved_bytes'] == 33
        checks['Q08_Q10_fsync_before_ledger_fault_injection'] = True

        data = bytes(range(256))*2048
        asset = upload(data)
        deadline = asset['expires_at']
        assert store.read_block(owner, asset['id'], 7, 257) == data[7:264]
        assert store.metadata(owner, asset['id'])['expires_at'] == deadline
        expect('asset_not_found', lambda: store.metadata(people[1][0], asset['id']))
        expect('asset_not_found', lambda: store.delete(people[1][0], asset['id']))
        checks['Q12_owner_boundary_Q19_download_no_renewal'] = True

        # Q09 Windows-like locked file: actual delete fails, no ledger release.
        before = store.usage(owner)
        target = store._path(asset['id'])
        original_unlink = Path.unlink
        def denied(path, *args, **kwargs):
            if path == target:
                raise PermissionError('Injected owned-file lock')
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', denied):
            assert store.delete(owner, asset['id'])['state'] == 'delete_failed'
        assert store.usage(owner)['used_bytes'] == before['used_bytes']
        assert target.is_file()
        assert store.delete(owner, asset['id'])['state'] == 'deleted'
        assert not target.exists()
        assert store.usage(owner)['used_bytes'] == before['used_bytes']-len(data)
        checks['Q05_direct_delete_Q09_failure_retains_bytes'] = True

        asset = upload(b'expired-input')
        with store._locked() as conn:
            conn.execute('UPDATE ptb_storage.assets SET expires_at=%s WHERE id=%s', (store._now(conn)-1, asset['id']))
        expect('asset_expired', lambda: store.read_block(owner, asset['id'], 0, 10))
        reopened.recover()
        assert not store._path(asset['id']).exists()
        checks['Q06_expired_chunk_rejected_Q15_recovery_removes_expired'] = True

        # Q16 no file or reservation gets accepted on a full server disk.
        before = store.usage(owner)
        with patch('ptb_api.storage.shutil.disk_usage', return_value=type('Disk', (), {'free': 0})()):
            expect('disk_space_low', lambda: create(1))
        assert store.usage(owner) == before
        checks['Q16_global_disk_guard'] = True

        # Real account adapter + HTTP boundaries; unique peer preserves old login buckets.
        settings = AuthSettings(origin='https://p07.test', signing_key=secrets.token_urlsafe(48))
        app = create_app(account_store=accounts, auth_settings=settings, storage=store)
        with TestClient(app, base_url=settings.origin, client=('p07-'+suffix, 51000)) as a, TestClient(app, base_url=settings.origin, client=('p07-'+suffix, 51001)) as b:
            headers = []
            for client, person in ((a, people[0]), (b, people[1])):
                csrf = client.get('/api/v1/auth/challenge').json()['csrf_token']
                result = client.post('/api/v1/auth/login', json={'username': person[2], 'password': person[3]}, headers={'Origin': settings.origin, 'X-CSRF-Token': csrf})
                assert result.status_code == 200
                headers.append({'Origin': settings.origin, 'X-CSRF-Token': result.json()['csrf_token'], 'X-PTB-Account': person[0]})
            asset = upload(b'abcdef')
            path = '/api/v1/assets/'+asset['id']
            assert a.get(path+'/content', headers={'Range': 'bytes=2-4'}).content == b'cde'
            assert b.get(path+'/content').status_code == 404
            assert b.delete(path, headers=headers[1]).status_code == 404
            assert a.delete(path).status_code == 403
            assert a.get(path+'/content', headers={'Range': 'bytes=6-'}).status_code == 416
        checks['real_pg_sessions_range_csrf_and_owner_http'] = True
    finally:
        # Only exact IDs created by this harness; no recursive delete and no SQL DELETE.
        for asset_id in created:
            store.delete(owner, asset_id)
    report = {'checks': checks, 'scope': 'Executed initial PG and disk cases; extended-validation.json covers contention, TCP and browser',
              'remaining_joint_gates': ['hard power-loss durability', 'ZIP and multi-file outputs',
                                       'P06 file fencing and active input references', 'sustained production load']}
    (ROOT/'output/validation/p07'/f'storage-{suffix}.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
