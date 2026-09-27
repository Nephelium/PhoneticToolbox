"""M04-E owned PostgreSQL, authenticated HTTP, Chrome and LPC worker. No DDL."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from uuid import uuid4
import urllib.request

import numpy as np
from scipy.io import wavfile

from ptb_api.account_store import PostgresAccountStore
from ptb_api.storage import Storage
from ptb_api.storage_policy import POLICY_VERSION, LEGACY_POLICY_VERSION
from ptb_worker.store import PostgresJobStore
from run_m01_validation import owned_postgres

ROOT = Path(__file__).resolve().parents[1]
HIDDEN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def verify(config, out):
    accounts = PostgresAccountStore(config['dsn'])
    storage = Storage(config['dsn'], ROOT / 'output/validation/p07/storage')
    jobs = PostgresJobStore(config['dsn'])
    with storage._locked() as conn:
        state = conn.execute('SELECT * FROM ptb_storage.state').fetchone()
        if state.get('policy_version', LEGACY_POLICY_VERSION) != POLICY_VERSION:
            raise RuntimeError('storage_policy_migration_required')
    storage.recover()
    with jobs.transaction(write=False) as tx:
        assert not tx.execute("SELECT 1 FROM {jobs} WHERE state IN ('queued','running','cancel_requested')").fetchone()
    people = []
    for _ in range(2):
        username = 'm04e_' + uuid4().hex[:12]
        password = secrets.token_urlsafe(32)
        user = accounts.create_user(username, password)
        project = accounts.create_project(str(user['id']), 'LPC 网页验证')
        people.append(dict(username=username, password=password, id=str(user['id']), project=str(project['id'])))
    inputs = out / 'inputs'
    inputs.mkdir()
    rate = 48000
    t = np.arange(rate, dtype=np.float64) / rate
    voiced = .3 * np.sin(2 * np.pi * 150 * t) + .1 * np.sin(2 * np.pi * 800 * t)
    wavfile.write(inputs / 'LPC ɑ̃˥.wav', rate, np.column_stack([voiced, voiced * .5]))
    wavfile.write(inputs / 'silent.wav', rate, np.zeros(rate, dtype=np.float64))
    grid = ('File type = "ooTextFile"\nObject class = "TextGrid"\n\n0\n1\n<exists>\n'
            '1\n"IntervalTier"\n"phones"\n0\n1\n2\n0\n.5\n"ɑ̃˥"\n.5\n1\n"末"\n')
    (inputs / 'LPC ɑ̃˥.TextGrid').write_text(grid, encoding='utf-8')
    os.environ['PTB_EGG_PYTHON'] = str(ROOT / '.venv/m03-compatible/python.exe')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    origin = f'http://127.0.0.1:{port}'
    options = dict(dsn=config['dsn'], storage_root=str(storage.root), enable_acoustic_batches=True)
    server = worker = None
    try:
        server = subprocess.Popen([sys.executable, '-m', 'ptb_api.server', '--port', str(port), '--frontend', str(ROOT / 'frontend/dist'), '--managed'],
                                  stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  text=True, encoding='utf-8', creationflags=HIDDEN)
        server.stdin.write(json.dumps(options | dict(signing_key=secrets.token_urlsafe(48), enable_jobs=True, enable_file_jobs=True)) + '\n')
        server.stdin.flush()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 20
        while True:
            try:
                with opener.open(origin + '/api/v1/health', timeout=1) as response:
                    assert json.load(response)['mode'] == 'server'
                break
            except OSError:
                if time.monotonic() > deadline or server.poll() is not None:
                    raise RuntimeError('Owned API unavailable')
                time.sleep(.1)
        worker = subprocess.Popen([sys.executable, '-m', 'ptb_worker.cli'], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                  encoding='utf-8', creationflags=HIDDEN)
        worker.stdin.write(json.dumps(options | dict(kind='postgres', reaper_binary=str(ROOT / 'phonetic_toolbox/core/acoustic/reaper.exe'))) + '\n')
        worker.stdin.flush()
        assert json.loads(worker.stdout.readline()) == {'ready': True}
        runtime = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
        settings = dict(origin=origin, people=people, playwright=str(runtime / 'node_modules/playwright'),
                        browser='C:/Program Files/Google/Chrome/Application/chrome.exe', output=str(out),
                        uploads=[str(inputs / name) for name in ('LPC ɑ̃˥.wav', 'LPC ɑ̃˥.TextGrid', 'silent.wav')])
        run = subprocess.run([str(runtime / 'bin/node.exe'), str(ROOT / 'tests/e2e/m04-web.cjs')],
                             input=json.dumps(settings, ensure_ascii=False) + '\n', capture_output=True,
                             text=True, encoding='utf-8', creationflags=HIDDEN, timeout=360)
        log = run.stdout + '\n' + run.stderr
        for person in people:
            log = log.replace(person['password'], '[redacted]')
        (out / 'browser.log').write_text(log, encoding='utf-8')
        assert run.returncode == 0, 'See browser.log'
        report = json.loads((out / 'web-report.json').read_text('utf-8'))
        for job_id in report['job_ids']:
            assert jobs.get(people[0]['id'], job_id)['state'] == 'succeeded'
        for item in report['downloads']:
            target = out / 'downloads' / item['saved']
            assert hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256']
        with storage._locked() as conn:
            for person in people:
                assert not conn.execute("SELECT 1 FROM ptb_storage.assets WHERE owner_id=%s AND kind='temporary' AND state!='deleted'", (person['id'],)).fetchone()
        return dict(browser_verified=True, authenticated_accounts=2, job_ids=report['job_ids'],
                    downloaded_files=len(report['downloads']), no_live_temporary_files=True)
    finally:
        for process in (worker, server):
            if process:
                if process.stdin:
                    process.stdin.close()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)


def main():
    out = ROOT / 'output/validation/m04-e' / ('web-' + uuid4().hex)
    out.mkdir(parents=True)
    report = dict(success=False, schema_applied=[], owned_postgres_stopped=False)
    try:
        with owned_postgres(out) as config:
            report.update(verify(config, out))
        report.update(success=True, owned_postgres_stopped=True)
    except RuntimeError as exc:
        report['blocked_by'] = str(exc)
        report['owned_postgres_stopped'] = True
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(out / 'report.json')


if __name__ == '__main__':
    main()
