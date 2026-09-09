"""P07 explicit initialization. Show is read-only; apply requires specific approval."""
import argparse
import json
import os
from pathlib import Path
import sys
from uuid import uuid4
import psycopg
from psycopg.conninfo import conninfo_to_dict
from ptb_api.storage import no_links

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / 'output/validation/p07/storage'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['show', 'apply'])
    parser.add_argument('--approved-p07-schema-and-test-files', action='store_true')
    args = parser.parse_args()
    sql = (ROOT/'backend/migrations/003_storage.sql').read_text('utf-8')
    if args.action == 'show':
        print(sql)
        return
    if not args.approved_p07_schema_and_test_files:
        parser.error('Requires specific approval of P07 tables and the isolated file root')
    config = json.loads(sys.stdin.readline())
    values = conninfo_to_dict(config['dsn'])
    if (values.get('host') != '127.0.0.1' or values.get('hostaddr', '127.0.0.1') != '127.0.0.1'
            or values.get('service') or STORAGE.exists()):
        parser.error('Loopback only; never replace an existing P07 root')
    STORAGE.parent.mkdir(parents=True, exist_ok=True)
    no_links(STORAGE.parent)
    with psycopg.connect(config['dsn'], autocommit=True, connect_timeout=5) as conn:
        if conn.execute('SELECT current_database()').fetchone()[0] != 'ptb_p05_test_20260909':
            parser.error('Unexpected database')
        if conn.execute("SELECT 1 FROM pg_namespace WHERE nspname='ptb_storage'").fetchone():
            parser.error('P07 already exists; initialization is never repeated')
        if conn.execute('SELECT version FROM ptb_accounts.schema_version').fetchall() != [(1,)]:
            parser.error('P05 prerequisite mismatch')
        if conn.execute('SELECT version FROM ptb_jobs.schema_version').fetchall() != [(1,)]:
            parser.error('P06 prerequisite mismatch')
        instance = uuid4()
        # A failure preserves any created evidence; no automatic DROP or directory removal.
        with conn.transaction():
            conn.execute(sql.replace('BEGIN;', '').replace('COMMIT;', ''))
            conn.execute('INSERT INTO ptb_storage.state(singleton,version,instance_id) VALUES(true,1,%s)', (instance,))
            STORAGE.mkdir(exist_ok=False)
            for name, data in [('.ptb-storage.json', json.dumps({'instance_id': str(instance)}).encode()),
                               ('.ptb-storage.lock', b'0')]:
                with (STORAGE/name).open('xb') as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
    print('P07 reviewed schema and marked isolated root initialized; P05/P06 tables preserved.')


if __name__ == '__main__':
    main()
