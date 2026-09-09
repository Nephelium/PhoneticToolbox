"""Explicit review/apply entry for 004, independent from the already approved 003."""
import argparse
import json
from pathlib import Path
import sys

from psycopg.conninfo import conninfo_to_dict
from ptb_api.storage import Storage

ROOT = Path(__file__).resolve().parents[1]
SQL = ROOT/'backend/migrations/004_job_assets.sql'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('show','apply'))
    parser.add_argument('--approved-p07-job-assets-schema', action='store_true')
    args = parser.parse_args()
    if args.action == 'show':
        print(SQL.read_text('utf-8'))
        return
    if not args.approved_p07_job_assets_schema:
        parser.error('Requires explicit review and approval of 004 job-file schema changes')
    config = json.loads(sys.stdin.readline())
    connection = conninfo_to_dict(config['dsn'])
    if connection.get('host') != '127.0.0.1' or connection.get('hostaddr', '127.0.0.1') != '127.0.0.1' or connection.get('service'):
        parser.error('Only the existing loopback test database is allowed')
    storage = Storage(config['dsn'], ROOT/'output/validation/p07/storage')
    with storage._locked() as conn:
        assert conn.execute('SELECT current_database() AS n').fetchone()['n'] == 'ptb_p05_test_20260909'
        assert conn.execute("SELECT to_regclass('ptb_storage.job_files_version') AS n").fetchone()['n'] is None
        for schema in ('ptb_accounts','ptb_jobs'):
            rows = conn.execute(f'SELECT version FROM {schema}.schema_version').fetchall()
            assert rows == [{'version':1}], 'Predecessor schema mismatch'
        constraint = conn.execute("""SELECT pg_get_constraintdef(oid) AS definition FROM pg_constraint
            WHERE conrelid='ptb_storage.assets'::regclass AND conname='assets_kind_check' AND contype='c'""").fetchone()
        assert constraint and constraint['definition'] == "CHECK ((kind = 'input'::text))", 'Unexpected resource-kind constraint'
        tables = ('ptb_accounts.users','ptb_accounts.sessions','ptb_accounts.projects','ptb_accounts.login_attempts',
                  'ptb_jobs.jobs','ptb_jobs.events','ptb_storage.state','ptb_storage.quota_accounts','ptb_storage.assets')
        def snapshot():
            return {table: sorted(json.dumps(row['data'],sort_keys=True,default=str) for row in
                conn.execute(f'SELECT to_jsonb(t) AS data FROM {table} t').fetchall()) for table in tables}
        with conn.transaction():
            conn.execute('SELECT pg_advisory_xact_lock(577606)')
            before = snapshot()
            text = SQL.read_text('utf-8')
            text = '\n'.join(line for line in text.splitlines() if line.strip() not in ('BEGIN;','COMMIT;'))
            conn.execute(text)
            assert snapshot() == before, 'Existing account/job/storage rows changed'
    print('004 job-file schema applied; pre-existing rows preserved. Joint capability still requires implementation and acceptance.')


if __name__ == '__main__':
    main()
