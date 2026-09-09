"""Own the existing isolated P05 PostgreSQL while applying/rechecking approved P06 tests."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import psycopg
from psycopg.conninfo import make_conninfo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/validation/p06'
DATA=ROOT/'output/validation/p05/postgres-data'
PRIVATE=ROOT/'output/validation/p05/postgres-private'
NO_WINDOW=getattr(subprocess,'CREATE_NO_WINDOW',0)


def run(args,config=None):
    result=subprocess.run([str(a) for a in args],input=json.dumps(config)+'\n' if config else None,
                          capture_output=True,text=True,encoding='utf-8',errors='replace',cwd=ROOT,
                          creationflags=NO_WINDOW,timeout=180)
    if result.returncode:
        raise RuntimeError(result.stdout+'\n'+result.stderr)
    print(result.stdout,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-p06-test-data',action='store_true')
    parser.add_argument('--apply-reviewed-schema',action='store_true')
    args=parser.parse_args()
    if not args.approved_p06_test_data:parser.error('Requires approval of the specific P06 test database scope')
    config=json.loads((PRIVATE/'connection.json').read_text('utf-8'))
    runtime=Path(config['runtime']).resolve()
    if Path(config['pgdata']).resolve()!=DATA.resolve() or not runtime.is_relative_to((ROOT/'.venv').resolve()):
        parser.error('Isolated runtime identity mismatch')
    if (DATA/'postmaster.pid').exists():parser.error('Owned test cluster must be stopped before this harness starts')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    config['dsn']=make_conninfo(config['dsn'],port=port)
    pg_ctl=runtime/'bin/pg_ctl.exe'
    OUT.mkdir(parents=True,exist_ok=True)
    started=False
    before={}
    try:
        with (OUT/'postgres-control.log').open('a',encoding='utf-8') as log:
            try:
                p=subprocess.run([str(pg_ctl),'-D',str(DATA),'-l',str(OUT/'postgres-server.log'),
                                  '-o',f'-p {port} -h 127.0.0.1','-w','-t','30','start'],
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=NO_WINDOW,timeout=45)
            finally:started=(DATA/'postmaster.pid').exists()
        if p.returncode:raise RuntimeError('P06 isolated PostgreSQL startup failed; see control log')
        with psycopg.connect(config['dsn']) as conn:
            assert conn.execute('SELECT current_database()').fetchone()[0]=='ptb_p05_test_20260909'
            assert conn.execute('SHOW listen_addresses').fetchone()[0]=='127.0.0.1'
            for table in ('users','sessions','projects','login_attempts','schema_version'):
                # Only retain a digest per pre-existing row, not credentials or payloads.
                for row in conn.execute(f'SELECT to_jsonb(t) FROM ptb_accounts.{table} t').fetchall():
                    raw=json.dumps(row[0],sort_keys=True,default=str)
                    before[(table,raw)]=hashlib.sha256(raw.encode()).hexdigest()
        (OUT/'p05-row-digests-before.json').write_text(json.dumps(sorted(before.values()),indent=2)+'\n',encoding='utf-8')
        if args.apply_reviewed_schema:
            run([sys.executable,'scripts/p06_database.py','apply','--kind','postgres','--approved-p06-test-schema'],config)
            run([sys.executable,'scripts/p06_database.py','apply','--kind','sqlite','--approved-p06-test-schema'])
        run([sys.executable,'scripts/verify_p06_jobs.py','--approved-test-data'],{'kind':'postgres','dsn':config['dsn']})
        run([sys.executable,'scripts/verify_p06_jobs.py','--approved-test-data'],{'kind':'sqlite','path':str(OUT/'local-state.sqlite3')})
        after=set()
        with psycopg.connect(config['dsn']) as conn:
            for table in ('users','sessions','projects','login_attempts','schema_version'):
                for row in conn.execute(f'SELECT to_jsonb(t) FROM ptb_accounts.{table} t').fetchall():
                    after.add((table,json.dumps(row[0],sort_keys=True,default=str)))
        assert set(before).issubset(after),'Pre-existing P05 rows changed'
    finally:
        if started:
            lines=(DATA/'postmaster.pid').read_text('utf-8').splitlines()
            if Path(lines[1]).resolve()!=DATA.resolve():raise RuntimeError('Refusing to stop unowned PGDATA')
            run([pg_ctl,'-D',DATA,'-w','-t','30','-m','fast','stop'])
    report={'scope':'approved P06 metadata task schema and test data','postgres_port':port,
            'p05_existing_rows_preserved':len(before),'postgres_stopped':True,
            'sql_sha256':{name:hashlib.sha256((ROOT/'backend/migrations'/name).read_bytes()).hexdigest()
                          for name in ('002_jobs.sql','002_jobs_sqlite.sql')}}
    (OUT/'database-validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
