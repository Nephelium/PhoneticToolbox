"""Own only the existing local test cluster; no schema or test-file actions without approval."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import psycopg
from psycopg.conninfo import make_conninfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/validation/p07'
DATA = ROOT/'output/validation/p05/postgres-data'
PRIVATE = ROOT/'output/validation/p05/postgres-private'
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def run(args, config=None):
    result = subprocess.run([str(x) for x in args], input=json.dumps(config)+'\n' if config else None,
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8', creationflags=NO_WINDOW, timeout=180)
    if result.returncode:
        # Keep failure details in this ignored test artifact, never the connection input.
        (OUT/'validation-failure.txt').write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
        raise RuntimeError('P07 validation failed; inspect local validation-failure.txt')
    print(result.stdout, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-p07-schema-and-test-files', action='store_true')
    parser.add_argument('--apply-reviewed-schema', action='store_true')
    parser.add_argument('--apply-approved-job-files-schema', action='store_true')
    args = parser.parse_args()
    if not args.approved_p07_schema_and_test_files:
        parser.error('Requires review of schema plus deletion of generated files in the marked P07 root')
    config = json.loads((PRIVATE/'connection.json').read_text('utf-8'))
    runtime = Path(config['runtime']).resolve()
    if Path(config['pgdata']).resolve() != DATA.resolve() or not runtime.is_relative_to((ROOT/'.venv').resolve()):
        parser.error('Test runtime mismatch')
    if (DATA/'postmaster.pid').exists():
        parser.error('Test cluster must be stopped before this harness starts')
    OUT.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    config['dsn'] = make_conninfo(config['dsn'], port=port)
    pg_ctl = runtime/'bin/pg_ctl.exe'; started = False; before = set()
    try:
        with (OUT/'postgres-control.log').open('a', encoding='utf-8') as log:
            try:
                process = subprocess.run([str(pg_ctl), '-D', str(DATA), '-l', str(OUT/'postgres-server.log'),
                    '-o', f'-p {port} -h 127.0.0.1', '-w', '-t', '30', 'start'], stdin=subprocess.DEVNULL,
                    stdout=log, stderr=log, creationflags=NO_WINDOW, timeout=45)
            finally:
                started = (DATA/'postmaster.pid').exists()
        if process.returncode:
            raise RuntimeError('Owned PostgreSQL failed to start')
        with psycopg.connect(config['dsn']) as conn:
            assert conn.execute('SELECT current_database()').fetchone()[0] == 'ptb_p05_test_20260909'
            for table in ('ptb_accounts.users', 'ptb_accounts.sessions', 'ptb_accounts.projects', 'ptb_accounts.login_attempts',
                          'ptb_accounts.schema_version', 'ptb_jobs.jobs', 'ptb_jobs.events', 'ptb_jobs.schema_version'):
                for row in conn.execute(f'SELECT to_jsonb(t) FROM {table} t').fetchall():
                    before.add((table, json.dumps(row[0], sort_keys=True, default=str)))
        if args.apply_reviewed_schema:
            run([sys.executable, '-X', 'utf8', 'scripts/p07_database.py', 'apply', '--approved-p07-schema-and-test-files'], config)
        if args.apply_approved_job_files_schema:
            run([sys.executable, '-X', 'utf8', 'scripts/p07_job_files_database.py', 'apply', '--approved-p07-job-assets-schema'], config)
        run([sys.executable, '-X', 'utf8', 'scripts/verify_p07_storage.py', '--approved-p07-schema-and-test-files'], config)
        run([sys.executable, '-X', 'utf8', 'scripts/verify_p07_extended.py', '--approved-p07-schema-and-test-files'], config)
        run([sys.executable, '-X', 'utf8', 'scripts/verify_p07_jobs.py', '--approved-p07-schema-and-test-files'], config)
        with psycopg.connect(config['dsn']) as conn:
            after = set()
            for table in ('ptb_accounts.users', 'ptb_accounts.sessions', 'ptb_accounts.projects', 'ptb_accounts.login_attempts',
                          'ptb_accounts.schema_version', 'ptb_jobs.jobs', 'ptb_jobs.events', 'ptb_jobs.schema_version'):
                for row in conn.execute(f'SELECT to_jsonb(t) FROM {table} t').fetchall():
                    after.add((table, json.dumps(row[0], sort_keys=True, default=str)))
        assert before.issubset(after), 'Pre-existing P05/P06 records changed'
    finally:
        if started:
            identity = (DATA/'postmaster.pid').read_text('utf-8').splitlines()
            assert Path(identity[1]).resolve() == DATA.resolve()
            run([pg_ctl, '-D', DATA, '-w', '-t', '30', '-m', 'fast', 'stop'])
    report = {'scope': 'P07 Windows controlled storage, file jobs and bounded ZIP joint acceptance',
              'existing_account_and_job_rows_preserved': len(before), 'postgres_stopped': True,
              'schema_sha256': hashlib.sha256((ROOT/'backend/migrations/003_storage.sql').read_bytes()).hexdigest(),
              'job_file_schema_sha256': hashlib.sha256((ROOT/'backend/migrations/004_job_assets.sql').read_bytes()).hexdigest()}
    (OUT/'database-validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
