"""Review/apply 005 only after explicit approval; no implicit initialization."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'output/validation/p06/local-state.sqlite3'
TABLES=('ptb_accounts.users','ptb_accounts.sessions','ptb_accounts.projects','ptb_accounts.login_attempts','ptb_accounts.schema_version',
        'ptb_jobs.jobs','ptb_jobs.events','ptb_jobs.schema_version','ptb_storage.state','ptb_storage.quota_accounts',
        'ptb_storage.assets','ptb_storage.job_assets','ptb_storage.job_files_version')


def sql_for(kind):
    return (ROOT/'backend/migrations'/('005_acoustic_batches.sql' if kind=='postgres' else '005_acoustic_batches_sqlite.sql')).read_text('utf-8')


def statements(sql):
    # Reviewed files contain no quoted semicolons or embedded procedural SQL.
    text='\n'.join(line for line in sql.splitlines() if not line.lstrip().startswith('--'))
    return [x.strip() for x in text.split(';') if x.strip() not in ('','BEGIN','BEGIN IMMEDIATE','COMMIT')]


def fingerprint(rows):
    return hashlib.sha256(json.dumps(sorted(json.dumps(dict(row),sort_keys=True,default=str) for row in rows),ensure_ascii=False).encode()).hexdigest()


def apply_sqlite(sql):
    from ptb_worker.io.scratch import no_links
    no_links(LOCAL)
    if not LOCAL.is_file():raise ValueError('Reviewed P06 database is missing; never create it implicitly')
    conn=sqlite3.connect(LOCAL.resolve().as_uri()+'?mode=rw',uri=True,timeout=5);conn.row_factory=sqlite3.Row
    try:
        conn.execute('PRAGMA foreign_keys=ON');conn.execute('BEGIN IMMEDIATE')
        if [dict(r) for r in conn.execute('SELECT version FROM schema_version')]!=[{'version':1}]:raise ValueError('P06 version mismatch')
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name IN ('acoustic_batch_version','acoustic_batches','acoustic_batch_items','m01_jobs_identity')").fetchone():raise ValueError('005 objects already exist')
        before={t:fingerprint(conn.execute('SELECT * FROM '+t).fetchall()) for t in ('jobs','events','schema_version')}
        for statement in statements(sql):conn.execute(statement)
        after={t:fingerprint(conn.execute('SELECT * FROM '+t).fetchall()) for t in before}
        if before!=after or conn.execute('PRAGMA foreign_key_check').fetchall():raise ValueError('Existing rows or FK integrity changed')
        conn.commit()
        return {'existing_tables_preserved':len(before),'new_batch_tables':3,'kind':'sqlite'}
    except BaseException:conn.rollback();raise
    finally:conn.close()


def apply_postgres(sql,config):
    from psycopg.conninfo import conninfo_to_dict
    from ptb_api.storage import Storage
    values=conninfo_to_dict(config['dsn'])
    if values.get('host')!='127.0.0.1' or values.get('hostaddr','127.0.0.1')!='127.0.0.1' or values.get('service'):
        raise ValueError('Only the reviewed loopback database is allowed')
    storage=Storage(config['dsn'],ROOT/'output/validation/p07/storage')
    with storage._locked() as conn,conn.transaction():
        conn.execute("SET LOCAL lock_timeout = '5s'");conn.execute("SET LOCAL statement_timeout = '30s'")
        if conn.execute('SELECT current_database() AS n').fetchone()['n']!='ptb_p05_test_20260909':raise ValueError('Unexpected database')
        conn.execute('SELECT pg_advisory_xact_lock(577606)')
        conn.execute('LOCK TABLE '+','.join(TABLES)+' IN SHARE ROW EXCLUSIVE MODE')
        for table in ('ptb_accounts.schema_version','ptb_jobs.schema_version','ptb_storage.job_files_version'):
            if conn.execute('SELECT version FROM '+table).fetchall()!=[{'version':1}]:raise ValueError('Predecessor version mismatch')
        for table in ('acoustic_batch_version','acoustic_batches','acoustic_batch_items'):
            if conn.execute('SELECT to_regclass(%s) AS n',('ptb_jobs.'+table,)).fetchone()['n'] is not None:raise ValueError('005 objects already exist')
        def snapshot():return {t:fingerprint(conn.execute('SELECT * FROM '+t).fetchall()) for t in TABLES}
        before=snapshot()
        for statement in statements(sql):conn.execute(statement)
        if snapshot()!=before:raise ValueError('Existing rows changed')
    return {'existing_tables_preserved':len(before),'new_batch_tables':3,'kind':'postgres'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('show','apply'))
    parser.add_argument('--kind',choices=('postgres','sqlite'),required=True)
    parser.add_argument('--approved-m01-batch-schema',action='store_true')
    parser.add_argument('--reviewed-sha256')
    args=parser.parse_args();sql=sql_for(args.kind);digest=hashlib.sha256(sql.encode()).hexdigest()
    if args.action=='show':print(json.dumps({'kind':args.kind,'sha256':digest}));print(sql);return
    if not args.approved_m01_batch_schema:parser.error('Explicit approval of docs/testing/m01-migration-review.md is required')
    if args.reviewed_sha256!=digest:parser.error('Review hash mismatch; inspect the exact SQL before applying')
    result=apply_postgres(sql,json.loads(sys.stdin.readline())) if args.kind=='postgres' else apply_sqlite(sql)
    print(json.dumps(result|{'schema_sha256':digest,'applied':True}))


if __name__=='__main__':main()
