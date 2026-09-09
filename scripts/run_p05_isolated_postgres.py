"""Approved Windows-only P05 empty-cluster validation; no installer, service or deletion.

The runtime must already have been downloaded and reviewed. Credentials stay in an
ACL-restricted ignored directory and private child stdin. This is a test harness,
not an application deployment or an automatic application-startup migration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

import httpx
import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from ptb_api.account_store import PostgresAccountStore

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output/validation/p05'
DATA = OUTPUT / 'postgres-data'
PRIVATE = OUTPUT / 'postgres-private'
DATABASE = 'ptb_p05_test_20260909'
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def run(args, *, input=None, check=True):
    result = subprocess.run([str(a) for a in args], input=input, text=True,
                            encoding='utf-8', errors='replace', capture_output=True,
                            creationflags=NO_WINDOW, cwd=ROOT, timeout=90)
    if check and result.returncode:
        # Do not include input, which may carry passwords or a private DSN.
        raise RuntimeError(f'{Path(args[0]).name} failed ({result.returncode}): {result.stderr}')
    return result


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--approved-empty-test-database', action='store_true')
    parser.add_argument('--resume-initialized-instance', action='store_true',
                        help='Resume this harness-owned stopped cluster after initialization; migration still requires an empty database')
    args = parser.parse_args()
    if not args.approved_empty_test_database:
        parser.error('Requires explicit prior approval for this dedicated empty test database')
    if os.name != 'nt':
        parser.error('This harness is for the reviewed Windows test runtime')
    runtime = args.runtime.resolve()
    if not runtime.is_relative_to((ROOT / '.venv').resolve()):
        parser.error('Runtime must be inside the isolated workspace .venv')
    if (DATA.exists() or PRIVATE.exists()) and not args.resume_initialized_instance:
        parser.error('Refusing to initialize over existing data or credentials')
    pg_ctl = runtime / 'bin/pg_ctl.exe'
    postgres = runtime / 'bin/postgres.exe'
    version = run([postgres, '--version']).stdout.strip()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not args.resume_initialized_instance:
        PRIVATE.mkdir()
    sid = run(['powershell.exe', '-NoProfile', '-Command',
               '[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value']).stdout.strip()
    if not sid.startswith('S-1-'):
        raise RuntimeError('Cannot determine current Windows identity')
    run(['icacls.exe', PRIVATE, '/inheritance:r', '/grant:r', f'*{sid}:(OI)(CI)F'])
    password = secrets.token_urlsafe(48)
    pwfile = PRIVATE / 'bootstrap-password.txt'
    if not args.resume_initialized_instance:
        pwfile.write_text(password + '\n', encoding='utf-8')
    pgport = free_port()
    admin_dsn = make_conninfo(host='127.0.0.1', port=pgport, user='ptb_p05_admin',
                             password=password, dbname='postgres', connect_timeout=5)
    dsn = make_conninfo(admin_dsn, dbname=DATABASE)
    config = {'dsn': dsn, 'signing_key': secrets.token_urlsafe(48), 'port': pgport,
              'runtime': str(runtime), 'pgdata': str(DATA)}
    if args.resume_initialized_instance:
        config = json.loads((PRIVATE / 'connection.json').read_text('utf-8'))
        if Path(config['runtime']).resolve() != runtime or Path(config['pgdata']).resolve() != DATA.resolve():
            parser.error('Stored test instance identity does not match')
        if (DATA / 'postmaster.pid').exists():
            parser.error('Resume requires the harness-owned instance to be stopped first')
        dsn, pgport = config['dsn'], config['port']
        connection = conninfo_to_dict(dsn)
        if (connection.get('host') != '127.0.0.1' or connection.get('dbname') != DATABASE
                or connection.get('user') != 'ptb_p05_admin' or connection.get('port') != str(pgport)
                or connection.get('hostaddr', '127.0.0.1') != '127.0.0.1'
                or connection.get('service')):
            parser.error('Stored connection is not the approved loopback test database')
        admin_dsn = make_conninfo(dsn, dbname='postgres')
    else:
        (PRIVATE / 'connection.json').write_text(json.dumps(config), encoding='utf-8')
        init = run([runtime / 'bin/initdb.exe', '-D', DATA, '-U', 'ptb_p05_admin',
                '--pwfile', pwfile, '--auth=scram-sha-256', '--encoding=UTF8',
                '--locale=C', '--data-checksums', '--no-clean',
                '-c', 'listen_addresses=127.0.0.1', '-c', f'port={pgport}',
                '-c', 'max_connections=40', '-c', 'shared_buffers=32MB'])
        (OUTPUT / 'postgres-initdb.log').write_text(init.stdout + init.stderr, encoding='utf-8')
    checks = {}
    started = False
    api = None

    def pg(action):
        nonlocal started
        if action == 'stop':
            # pg_ctl targets this PGDATA; verify the live pid file names our directory.
            lines = (DATA / 'postmaster.pid').read_text('utf-8').splitlines()
            if Path(lines[1]).resolve() != DATA.resolve():
                raise RuntimeError('Refusing to stop an unowned PostgreSQL data directory')
            run([pg_ctl, '-D', DATA, '-w', '-t', '30', '-m', 'fast', 'stop'])
            started = False
        else:
            # Windows descendants can retain anonymous pipe handles even after
            # pg_ctl exits. Use a file, and wait only on our pg_ctl process.
            with (OUTPUT / 'postgres-control.log').open('a', encoding='utf-8') as log:
                try:
                    result = subprocess.run([str(pg_ctl), '-D', str(DATA), '-l',
                                             str(OUTPUT / 'postgres-server.log'), '-w', '-t', '30', 'start'],
                                            stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                            creationflags=NO_WINDOW, timeout=45)
                finally:
                    started = (DATA / 'postmaster.pid').exists()
            if result.returncode:
                raise RuntimeError('Owned PostgreSQL startup failed; inspect control log')

    def start_api(port):
        process = subprocess.Popen([sys.executable, '-m', 'ptb_api.server', '--port', str(port),
                                    '--frontend', str(ROOT / 'frontend/dist')],
                                   stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, text=True, encoding='utf-8',
                                   creationflags=NO_WINDOW, cwd=ROOT)
        process.stdin.write(json.dumps(config) + '\n')
        process.stdin.close()
        return process

    def wait_api(client, process):
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError('Owned P05 API process failed to start')
            try:
                if client.get('/api/v1/auth/challenge').status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(0.1)
        raise RuntimeError('Owned P05 API did not become ready')

    try:
        pg('start')
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            assert conn.execute('SHOW listen_addresses').fetchone()[0] == '127.0.0.1'
            assert conn.execute('SHOW password_encryption').fetchone()[0] == 'scram-sha-256'
            conn.execute('CREATE DATABASE ptb_p05_test_20260909')
        checks['loopback_and_scram'] = True
        migration_args = [sys.executable, ROOT / 'scripts/p05_database.py', 'apply',
                          '--approved-empty-test-database', '--dsn-stdin']
        print(run(migration_args, input=dsn + '\n').stdout, flush=True)
        # A second application must refuse the now nonempty database.
        refused = run(migration_args, input=dsn + '\n', check=False)
        assert refused.returncode != 0 and 'non-empty database' in refused.stderr
        checks['migration_and_nonempty_refusal'] = True
        result = run([sys.executable, ROOT / 'scripts/verify_p05_postgres.py',
                      '--approved-test-data', '--dsn-stdin'], input=dsn + '\n')
        print(result.stdout, flush=True)
        checks['postgres_five_checks'] = True

        # Additional end-to-end evidence with a real socket and new API/PG processes.
        store = PostgresAccountStore(dsn)
        username, user_password = 'restart_' + secrets.token_hex(6), secrets.token_urlsafe(32)
        store.create_user(username, user_password)
        apiport = free_port()
        origin = f'http://127.0.0.1:{apiport}'
        api = start_api(apiport)
        with httpx.Client(base_url=origin, timeout=10, trust_env=False) as client:
            wait_api(client, api)
            assert client.get('/server/').status_code == 200
            csrf = client.get('/api/v1/auth/challenge').json()['csrf_token']
            login = client.post('/api/v1/auth/login', json={'username': username, 'password': user_password},
                                headers={'Origin': origin, 'X-CSRF-Token': csrf})
            assert login.status_code == 200
            headers = {'Origin': origin, 'X-CSRF-Token': login.json()['csrf_token'],
                       'X-PTB-Account': login.json()['user']['id']}
            project = client.post('/api/v1/projects', json={'name': '重启后仍保留的中文项目'}, headers=headers)
            assert project.status_code == 201
            project_id = project.json()['id']
            api.terminate()  # Exact Popen handle created above, never a port/name search.
            api.wait(timeout=15)
            api = None
            pg('stop')
            pg('start')
            api = start_api(apiport)
            wait_api(client, api)
            assert client.get('/api/v1/auth/me').status_code == 200
            assert client.get('/api/v1/projects/' + project_id).json()['name'] == '重启后仍保留的中文项目'
            checks['api_and_database_process_restart_restore'] = True
            cookie = client.cookies.get('ptb-dev-session')
            assert client.post('/api/v1/auth/logout', headers=headers).status_code == 204
            with httpx.Client(base_url=origin, trust_env=False, cookies={'ptb-dev-session': cookie}) as old:
                assert old.get('/api/v1/auth/me').status_code == 401
            checks['real_http_logout_revocation'] = True
    finally:
        if api is not None and api.poll() is None:
            api.terminate()
            api.wait(timeout=15)
        if started:
            pg('stop')
    checks['owned_processes_stopped'] = True
    report = {'version': version, 'database': DATABASE, 'port': pgport,
              'sql_sha256': hashlib.sha256((ROOT / 'backend/migrations/001_accounts.sql').read_bytes()).hexdigest(),
              'checks': checks, 'scope': 'Windows isolated PostgreSQL and loopback HTTP; no crash recovery, public TLS or P06/P07 evidence'}
    (OUTPUT / 'postgres-integration.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
