"""Authorized generated-file tests: real PG, quota contention, TCP and browser.

No schema initialization. Every file removed here belongs to a newly created
test account under the pre-marked P07 root. Never delete directories or old rows.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time
from uuid import uuid4

import httpx
import uvicorn
from starlette.staticfiles import StaticFiles
from ptb_api.account_store import PostgresAccountStore
from ptb_api.auth import AuthSettings
from ptb_api.main import create_app
from ptb_api.quota import CHUNK_BYTES, QUOTA_BYTES, StorageError
from ptb_api.storage import Storage
from ptb_api.storage_models import UploadInput
from ptb_worker.cleanup import run_cleanup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/validation/p07'
FILES = OUT/'storage'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-p07-schema-and-test-files', action='store_true')
    if not parser.parse_args().approved_p07_schema_and_test_files:
        parser.error('Requires reviewed test database and generated-file deletion approval')
    config = json.loads(sys.stdin.readline())
    accounts = PostgresAccountStore(config['dsn'])
    with accounts.connection() as conn:
        assert conn.execute('SELECT current_database() AS n').fetchone()['n'] == 'ptb_p05_test_20260909'
    storage = Storage(config['dsn'], FILES, min_free_bytes=0)
    storage.recover()
    suffix = uuid4().hex[:10]
    people, checks = [], {}
    for i in range(10):
        username, password = f'p07x_{suffix}_{i}', secrets.token_urlsafe(32)
        user = accounts.create_user(username, password)
        project = accounts.create_project(str(user['id']), 'P07 真实存储联合检查')
        people.append(dict(owner=str(user['id']), project=str(project['id']), username=username, password=password))
    owner, project = people[0]['owner'], people[0]['project']

    def create(person, size=None, name='合成 ɑ.bin'):
        return storage.create(person['owner'], UploadInput(project_id=person['project'],
            name=name, expected_bytes=size, idempotency_key=uuid4().hex))

    def upload(person, data, name='合成 ɑ.bin'):
        asset = create(person, len(data), name)
        for offset in range(0, len(data), CHUNK_BYTES):
            storage.append(person['owner'], asset['id'], offset, data[offset:offset+CHUNK_BYTES])
        return storage.finalize(person['owner'], asset['id'], hashlib.sha256(data).hexdigest())

    def expect(code, action):
        try:
            action()
        except StorageError as error:
            assert error.code == code, error.code
        else:
            raise AssertionError('Expected '+code)

    server = None
    server_thread = None
    cleanup_stop = threading.Event()
    cleanup_thread = None
    resume = threading.Event()
    try:
        downloadable = upload(people[0], b'full-account-download')
        fill = create(people[0], QUOTA_BYTES-len(b'full-account-download')-1)
        barrier = threading.Barrier(4)
        def contend(_):
            # Independent Storage objects exercise the same OS and PG locks.
            other = Storage(config['dsn'], FILES, min_free_bytes=0)
            other.ready = True
            barrier.wait(timeout=10)
            try:
                return other.create(owner, UploadInput(project_id=project, name='last-byte.bin',
                    expected_bytes=1, idempotency_key=uuid4().hex))['id']
            except StorageError as error:
                return error.code
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(contend, range(4)))
        assert outcomes.count('quota_exceeded') == 3, outcomes
        last_id = next(value for value in outcomes if value != 'quota_exceeded')
        assert storage.usage(owner)['available_bytes'] == 0
        expect('quota_exceeded', lambda: create(people[0], 1))
        checks['Q01_last_byte_four_contenders'] = {'accepted': 1, 'rejected': 3, 'quota_bytes': QUOTA_BYTES}

        # The server uses the real account/storage adapters. Only the test peer
        # label is unique, preserving pre-existing P05 IP throttle records.
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        origin = f'http://127.0.0.1:{port}'
        app = create_app(account_store=accounts, storage=storage, auth_settings=AuthSettings(
            origin=origin, signing_key=secrets.token_urlsafe(48), allow_insecure_loopback=True))
        app.mount('/server', StaticFiles(directory=ROOT/'frontend/dist', html=True))
        async def test_peer(scope, receive, send):
            if scope['type'] == 'http':
                assert scope['client'][0] == '127.0.0.1'
                scope = dict(scope, client=('p07x-'+suffix, scope['client'][1]))
            await app(scope, receive, send)
        server = uvicorn.Server(uvicorn.Config(test_peer, access_log=False, log_level='critical', proxy_headers=False))
        server_thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
        server_thread.start()
        deadline = time.monotonic()+10
        while not server.started and server_thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.05)
        assert server.started, 'Owned TCP host did not start'
        with httpx.Client(base_url=origin, timeout=20) as client:
            challenge = client.get('/api/v1/auth/challenge').json()['csrf_token']
            login = client.post('/api/v1/auth/login', json={key: people[0][key] for key in ('username','password')},
                headers={'Origin': origin, 'X-CSRF-Token': challenge})
            assert login.status_code == 200
            headers = {'Origin': origin, 'X-CSRF-Token': login.json()['csrf_token'], 'X-PTB-Account': owner}
            content = '/api/v1/assets/'+downloadable['id']+'/content'
            assert client.get(content).content == b'full-account-download'
            deleted = client.delete('/api/v1/assets/'+last_id, headers=headers)
            assert deleted.status_code == 200 and deleted.json()['state'] == 'deleted'
            assert storage.usage(owner)['available_bytes'] == 1
            storage.delete(owner, fill['id'])
            checks['Q13_full_account_login_download_delete_tcp'] = True

            # A real TCP transfer with a deterministic server-side pause before
            # the second block. Expire this test row after the first received
            # block, then verify the response truncates rather than sending it.
            data = bytes(range(256))*(CHUNK_BYTES*3//256)
            expiring = upload(people[0], data)
            second = threading.Event()
            original_read = storage.read_block
            def paused_read(who, asset_id, offset, size):
                if str(asset_id) == expiring['id'] and offset >= CHUNK_BYTES:
                    second.set()
                    assert resume.wait(10), 'Test did not release second-block pause'
                return original_read(who, asset_id, offset, size)
            storage.read_block = paused_read
            received, truncated = bytearray(), False
            try:
                with client.stream('GET', '/api/v1/assets/'+expiring['id']+'/content') as response:
                    assert response.status_code == 200
                    try:
                        for block in response.iter_bytes(CHUNK_BYTES):
                            received.extend(block)
                            if len(received) == CHUNK_BYTES:
                                assert second.wait(5)
                                with storage._locked() as conn:
                                    conn.execute('UPDATE ptb_storage.assets SET expires_at=%s WHERE id=%s',
                                        (storage._now(conn)-.001, expiring['id']))
                                resume.set()
                    except httpx.RemoteProtocolError:
                        truncated = True
                assert truncated and received == data[:CHUNK_BYTES]
                assert client.get('/api/v1/assets/'+expiring['id']+'/content').status_code == 410
                assert client.get('/api/v1/assets/'+expiring['id']+'/content', headers={'Range':'bytes=1-8'}).status_code == 410
            finally:
                resume.set()
                storage.read_block = original_read
            checks['Q06_tcp_expiry_after_first_block'] = {'sent_bytes':len(received), 'total_bytes':len(data), 'truncated':truncated,
                'fixture':'controlled server-side pause before second block; real TCP, PG deadline and storage reads'}

        # Ten real accounts perform controlled uploads and download verification.
        # This is bounded storage concurrency, not a sustained production load.
        def account_work(person):
            data = bytes([people.index(person)])*CHUNK_BYTES
            asset = upload(person, data)
            assert storage.read_block(person['owner'], asset['id'], 0, CHUNK_BYTES) == data
            assert storage.metadata(person['owner'], asset['id'])['sha256'] == hashlib.sha256(data).hexdigest()
            return asset
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=10) as pool:
            assets = list(pool.map(account_work, people))
        checks['ten_accounts_concurrent_storage'] = {'accounts':10,'bytes_each':CHUNK_BYTES,'seconds':round(time.monotonic()-started,3)}
        for person, asset in zip(people, assets):
            stranger = people[(people.index(person)+1)%10]
            expect('asset_not_found', lambda: storage.metadata(stranger['owner'], asset['id']))

        # Q18 competing cleanup/direct deletion must release the charge once.
        raced = upload(people[0], b'cleanup-delete-race')
        before = storage.usage(owner)
        with storage._locked() as conn:
            conn.execute('UPDATE ptb_storage.assets SET expires_at=%s WHERE id=%s', (storage._now(conn)-1, raced['id']))
        # Remove the earlier expired transfer so the accounting delta is exact.
        storage.delete(owner, expiring['id'])
        before = storage.usage(owner)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(storage.cleanup), pool.submit(storage.delete, owner, raced['id']),
                       pool.submit(storage.delete, owner, raced['id'])]
            for future in futures:
                future.result()
        after = storage.usage(owner)
        assert before['used_bytes']-after['used_bytes'] == len(b'cleanup-delete-race')
        assert before['reserved_bytes'] == after['reserved_bytes']
        checks['Q18_cleanup_and_two_deletes_charge_once'] = True

        # Measure the real worker loop; it must not delete before the deadline.
        expiring = upload(people[0], b'actual-cleanup-deadline')
        with storage._locked() as conn:
            due = storage._now(conn)+1.5
            conn.execute('UPDATE ptb_storage.assets SET expires_at=%s WHERE id=%s', (due, expiring['id']))
        cleanup_thread = threading.Thread(target=run_cleanup, args=(storage,cleanup_stop), daemon=True)
        cleanup_thread.start()
        time.sleep(.2)
        assert storage.metadata(owner, expiring['id'])['state'] == 'ready'
        timeout = time.monotonic()+6
        while storage._path(expiring['id']).exists() and time.monotonic()<timeout:
            time.sleep(.05)
        assert not storage._path(expiring['id']).exists()
        with storage._locked() as conn:
            deleted_at = storage._row(conn, owner, expiring['id'])['deleted_at']
        assert 0 <= deleted_at-due < 2
        checks['cleanup_deadline_measured'] = {'late_seconds':round(deleted_at-due,4),'not_deleted_early':True}
        cleanup_stop.set()
        cleanup_thread.join(10)
        assert not cleanup_thread.is_alive()
        (OUT/'extended-before-browser.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
        print('Real PG quota contention, TCP expiry, ten-account storage and cleanup checks passed.', flush=True)

        runtime = Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
        browser_input = dict(origin=origin, people=[people[8],people[9]],
            playwright=str(runtime/'node_modules/playwright'), browser='C:/Program Files/Google/Chrome/Application/chrome.exe',
            output=str(ROOT/'output/playwright/p07'))
        result = subprocess.run([str(runtime/'bin/node.exe'), 'tests/e2e/p07-storage.cjs'], input=json.dumps(browser_input)+'\n',
            text=True, encoding='utf-8', capture_output=True, cwd=ROOT, timeout=90,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (OUT/'browser-run.log').write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
        assert result.returncode == 0, 'See local browser-run.log'
        checks['real_pg_storage_browser'] = True
    finally:
        resume.set()
        cleanup_stop.set()
        if cleanup_thread is not None:
            cleanup_thread.join(10)
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(15)
            assert not server_thread.is_alive(), 'Owned TCP host did not stop'
        # Only the ten new test accounts and their registered UUID resources.
        for person in people:
            for asset in storage.list(person['owner'], person['project']):
                assert storage._path(asset['id']).resolve().parent == FILES.resolve()
                assert storage.delete(person['owner'], asset['id'])['state'] == 'deleted'
            usage = storage.usage(person['owner'])
            assert usage['used_bytes'] == usage['reserved_bytes'] == 0
    report = dict(scope='Windows PG and controlled single-file storage acceptance', checks=checks,
        owned_host_stopped=True, generated_files_removed=True,
        remaining=['ZIP and result generation with P06 fencing/input references','hard power-cut and cross-platform validation','sustained production load'])
    (OUT/'extended-validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
