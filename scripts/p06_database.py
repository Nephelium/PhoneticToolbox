"""Review-first P06 task schema tool. No automatic startup migration."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import psycopg
from psycopg.conninfo import conninfo_to_dict

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'output/validation/p06/local-state.sqlite3'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['show','apply'])
    parser.add_argument('--kind',choices=['postgres','sqlite'],required=True)
    parser.add_argument('--approved-p06-test-schema',action='store_true')
    args=parser.parse_args()
    sql=(ROOT/'backend/migrations'/('002_jobs.sql' if args.kind=='postgres' else '002_jobs_sqlite.sql')).read_text('utf-8')
    if args.action=='show':print(sql);return
    if not args.approved_p06_test_schema:parser.error('P06 schema requires separate explicit approval')
    if args.kind=='postgres':
        config=json.loads(sys.stdin.readline());connection=conninfo_to_dict(config['dsn'])
        if connection.get('host')!='127.0.0.1' or connection.get('hostaddr','127.0.0.1')!='127.0.0.1' or connection.get('service'):
            parser.error('Only the reviewed loopback test instance is supported')
        with psycopg.connect(config['dsn'],autocommit=True,connect_timeout=5) as conn:
            if conn.execute('SELECT current_database()').fetchone()[0]!='ptb_p05_test_20260909':
                parser.error('Expected the approved dedicated P05 test database')
            if conn.execute('SELECT version FROM ptb_accounts.schema_version').fetchall()!=[(1,)]:
                parser.error('P05 prerequisite mismatch')
            if conn.execute("SELECT 1 FROM pg_namespace WHERE nspname='ptb_jobs'").fetchone():
                parser.error('P06 schema already exists; refusing overwrite')
            conn.execute(sql)
    else:
        LOCAL.parent.mkdir(parents=True,exist_ok=True)
        # Exclusive creation before SQLite opens it; an existing file is never reused.
        with LOCAL.open('xb'):pass
        with sqlite3.connect(LOCAL) as conn:
            conn.execute('PRAGMA foreign_keys=ON');conn.executescript(sql)
    print('Approved P06 '+args.kind+' schema applied; existing P05 data preserved')


if __name__=='__main__':main()
