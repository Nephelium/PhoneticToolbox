"""Real browser -> loopback HTTP -> fresh PG -> existing scientific worker.

Opt in; no SMTP, production database, desktop cache or installed EXE is used.
"""
import hashlib
import json
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn
from scipy.io import wavfile
from starlette.staticfiles import StaticFiles

from ptb_api.auth import AuthSettings
from ptb_api.main import create_app
from ptb_api.registration import RegistrationService
from ptb_api.registration_store import PostgresRegistrationStore
from ptb_api.workspaces import WorkspaceService
from ptb_worker.acoustic_batches import AcousticBatches
from ptb_worker.acoustic_files import AcousticFiles
from ptb_worker.executor import run_worker
from ptb_worker.store import PostgresJobStore
from test_p07_policy_postgres import legacy, migrate, MIGRATIONS, ROOT
from web_postgres_support import cluster, artifacts


@pytest.mark.parametrize('browser', ['chrome', 'edge'])
def test_real_browser_science_and_logout(legacy, monkeypatch, browser):
    e = legacy
    migrate(e)
    for name in ('008_web_identity.sql', '009_web_workspaces.sql','010_web_capacity.sql'):
        e.conn.execute((MIGRATIONS / name).read_text('utf-8'))
    monkeypatch.setenv('PTB_EGG_PYTHON', str(ROOT / '.venv/m03-compatible/python.exe'))
    # The worker changes cwd into its owned scratch; relative PYTHONPATH entries
    # from a test invocation must not redirect its scientific child imports.
    import os
    monkeypatch.setenv('PYTHONPATH',os.pathsep.join(str(ROOT/p) for p in ('backend/src','packages/phonetic_core/src')))
    e.workspaces = WorkspaceService(e.storage)
    e.workspaces.activate()
    jobs = PostgresJobStore(e.dsn, max_running=1)
    files = AcousticFiles(jobs, e.storage)
    files.reaper_binary = str(ROOT / 'phonetic_toolbox/core/acoustic/reaper.exe')
    AcousticBatches(jobs, files)
    with artifacts.verification_run('web-launch', 'compute-' + browser,
            'Visible browser, real HTTP/PG/native science; EGG-SYN-PCM16.npz; synthetic mail captured locally') as (out, scratch):
        from ptb_worker import acoustic_executor
        original_error = acoustic_executor.public_error
        def diagnose(error):
            import traceback
            (out / 'worker-error.txt').write_text(''.join(traceback.format_exception(error)), encoding='utf-8')
            return original_error(error)
        monkeypatch.setattr(acoustic_executor, 'public_error', diagnose)
        original_claim = jobs.claim
        # A test-owned pause before claiming makes queued cancellation deterministic;
        # science execution and all task state transitions remain the real code.
        monkeypatch.setattr(jobs, 'claim', lambda worker: None if (scratch / 'pause-worker').exists() else original_claim(worker))
        with np.load(ROOT / 'tests/fixtures/m03/EGG-SYN-PCM16.npz') as data:
            samples = np.column_stack([data['load.egg_signal_raw'], data['load.audio_signal']])
        wavfile.write(scratch / 'synthetic.wav', 44100, samples)
        class MailCapture:
            def send(self, recipient, subject, body, key):
                token = re.search(r'#(?:verify|reset)=([A-Za-z0-9_-]{43})', body)[1]
                (scratch / 'mail-token.txt').write_text(token, encoding='ascii')
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        origin = 'http://127.0.0.1:' + str(sock.getsockname()[1])
        accounts = PostgresRegistrationStore(e.dsn)
        app = create_app(account_store=accounts, auth_settings=AuthSettings(origin=origin,
            signing_key='isolated-compute-check-' * 3, allow_insecure_loopback=True),
            job_store=jobs, storage=e.storage, workspace_service=e.workspaces,
            registration_service=RegistrationService(accounts, MailCapture(), origin))
        app.mount('/server', StaticFiles(directory=ROOT / 'output/validation/web-launch/frontend', html=True))
        server = uvicorn.Server(uvicorn.Config(app, log_level='warning', access_log=False, proxy_headers=False))
        http = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
        stop = threading.Event()
        worker = threading.Thread(target=run_worker, args=(jobs, stop, 'web-flow'), daemon=True)
        approval_errors=[]
        def approve_capacity():
            try:
                while not stop.wait(.1):
                    if not (scratch/'capacity-approve').exists():continue
                    from ptb_api.web_capacity import decide
                    with e.workspaces._read() as c:
                        row=c.execute("SELECT id FROM ptb_web.capacity_requests WHERE state='pending'").fetchone()
                    assert row
                    decide(e.workspaces,row['id'],granted_bytes=1_500_000_000,duration='7d',note='已按本次处理需求批准7天。')
                    return
            except Exception as error:approval_errors.append(str(error))
        approver=threading.Thread(target=approve_capacity,daemon=True)
        http.start()
        worker.start()
        approver.start()
        try:
            until = time.monotonic() + 15
            while not server.started:
                assert http.is_alive() and time.monotonic() < until
                time.sleep(.02)
            with (out / 'browser.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(['node', str(ROOT / 'tests/e2e/web-compute.cjs'),
                    str(out), str(scratch), origin, browser], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            assert result.returncode == 0, (out / 'browser.log').read_text('utf-8')
            assert not approval_errors
            report = json.loads((out / 'checks.json').read_text('utf-8'))
            # Independently check returned bytes, source selection and scientific output structure.
            for job in report['jobs']:
                for item in job['result_manifest']['files']:
                    raw = (scratch / 'results' / item['id']).read_bytes()
                    assert len(raw) == item['size_bytes']
                    assert hashlib.sha256(raw).hexdigest() == item['sha256']
                by_name = {f['name']: scratch / 'results' / f['id'] for f in job['result_manifest']['files']}
                if job['operation'] == 'lpc_analysis':
                    meta = json.loads(by_name['lpc.ptb.json'].read_text('utf-8'))
                    rate, audio = wavfile.read(by_name['lpc_AUDIO.wav'])
                    a, b = meta['selection']['start_sample'], meta['selection']['end_sample']
                    assert rate == 44100 and np.array_equal(audio, samples[a:b].astype(float).mean(axis=1))
                elif job['operation'] == 'egg_analysis':
                    import pandas as pd
                    frame = pd.read_csv(by_name['egg_DATA.csv'])
                    assert len(frame) > 0 and frame['CQ'].notna().any()
                elif job['operation'] == 'acoustic_analysis':
                    from ptb_api.acoustic_models import AcousticResult
                    value = AcousticResult.model_validate_json(by_name['result.ptb.json'].read_bytes())
                    assert any(b.actual == 'native_reaper' for b in value.metadata.backends)
                    assert next(c for c in value.numeric if c.key == 'rF0').nonfinite.count(0) > 0
            e.storage.cleanup()
            user = accounts.get_user('browser_' + browser)
            assert user is not None
            for table in ('ptb_accounts.projects', 'ptb_jobs.jobs', 'ptb_storage.assets'):
                assert e.conn.execute('SELECT count(*) AS n FROM ' + table + ' WHERE owner_id=%s', (user['id'],)).fetchone()['n'] == 0
            assert e.storage.usage(str(user['id']))['used_bytes'] == 0
            report.update(scientific_readback=True, logout_database_and_quota_clean=True)
            artifacts.write_json(out / 'checks.json', report)
        finally:
            stop.set()
            worker.join(30)
            approver.join(2)
            server.should_exit = True
            http.join(15)
            sock.close()
            assert not worker.is_alive() and not http.is_alive()


def test_preview_entry_owns_worker_and_stops_on_stdin_close(legacy, monkeypatch):
    """Exercise the actual launcher entry and opt-in worker, not only create_app."""
    import os
    import sys
    import httpx
    from uuid import uuid4
    e=legacy;migrate(e)
    for name in ('008_web_identity.sql','009_web_workspaces.sql','010_web_capacity.sql'):
        e.conn.execute((MIGRATIONS/name).read_text('utf-8'))
    accounts=PostgresRegistrationStore(e.dsn)
    accounts.create_user('preview_check','synthetic-password-only')
    monkeypatch.setenv('PTB_EGG_PYTHON',str(ROOT/'.venv/m03-compatible/python.exe'))
    with artifacts.verification_run('web-launch','preview-entry',
            'Real server.py managed subprocess, own PG, LPC and EOF shutdown') as (out,scratch):
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        config=dict(dsn=e.dsn,signing_key='preview-synthetic-only-'*3,storage_root=str(e.root),
            enable_jobs=True,enable_file_jobs=True,enable_acoustic_batches=True,
            enable_web_workspaces=True,enable_worker=True,
            reaper_binary=str(ROOT/'phonetic_toolbox/core/acoustic/reaper.exe'))
        with (out/'server.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-B','-m','ptb_api.server','--managed','--port',str(port),
                '--frontend',str(ROOT/'output/validation/web-launch/frontend')],stdin=subprocess.PIPE,
                stdout=log,stderr=log,text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                child.stdin.write(json.dumps(config)+'\n');child.stdin.flush()
                origin=f'http://127.0.0.1:{port}'
                with httpx.Client(base_url=origin,trust_env=False,timeout=10) as client:
                    deadline=time.monotonic()+20
                    while True:
                        assert child.poll() is None and time.monotonic()<deadline
                        try:
                            if client.get('/api/v1/health').status_code==200:break
                        except httpx.TransportError:pass
                        time.sleep(.1)
                    assert client.get('/server/').status_code==200
                    csrf=client.get('/api/v1/auth/challenge').json()['csrf_token']
                    client.headers.update({'Origin':origin,'X-CSRF-Token':csrf})
                    session=client.post('/api/v1/auth/login',json=dict(username='preview_check',password='synthetic-password-only')).json()
                    client.headers.update({'X-CSRF-Token':session['csrf_token'],'X-PTB-Account':session['user']['id']})
                    assert client.post('/api/v1/workspaces/open',json={}).status_code==200
                    project=client.post('/api/v1/projects',json={'name':'preview'}).json()['id']
                    import io
                    source=io.BytesIO();wavfile.write(source,16000,np.random.default_rng(7).normal(0,.1,8000));raw=source.getvalue()
                    a=client.post('/api/v1/uploads',json=dict(project_id=project,name='synthetic.wav',expected_bytes=len(raw),idempotency_key=uuid4().hex)).json()
                    assert client.put('/api/v1/uploads/'+a['id']+'/blocks?offset=0',content=raw,headers={'Content-Type':'application/octet-stream'}).is_success
                    a=client.post('/api/v1/uploads/'+a['id']+'/finalize',json={'sha256':hashlib.sha256(raw).hexdigest()}).json()
                    r=client.post('/api/v1/jobs/lpc/create',json=dict(project_id=project,idempotency_key=uuid4().hex,
                        audio=dict(asset_id=a['id'],sha256=a['sha256']),config={'roi_end':.05}))
                    assert r.status_code==201,r.text
                    job=r.json();deadline=time.monotonic()+45
                    while job['state'] in ('queued','running','cancel_requested'):
                        assert time.monotonic()<deadline
                        time.sleep(.1);job=client.get('/api/v1/jobs/'+job['id']).json()
                    assert job['state']=='succeeded',job
                    assert client.post('/api/v1/auth/logout').status_code==204
                child.stdin.close();assert child.wait(timeout=30)==0
                artifacts.write_json(out/'report.json',dict(actual_preview_entry=True,science='lpc_analysis',state=job['state'],managed_shutdown=True))
            finally:
                if child.stdin and not child.stdin.closed:child.stdin.close()
                if child.poll() is None:
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=10)
