"""M01 owned local PostgreSQL harness; approved 005 migration and synthetic tests."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from uuid import uuid4
from psycopg.conninfo import make_conninfo
from ptb_worker.io.scratch import no_links

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'output/validation/p05/postgres-data'
PRIVATE=ROOT/'output/validation/p05/postgres-private/connection.json'
HASHES={'postgres':'3d1006b267d3db69d6e0b9bc3419dd7062e261aba24f21712c5bf7ec169dedf6',
        'sqlite':'d66c08a7885f78676651afe1f08c0c9e459aafae901fc4bfe5fde8d8e6c47847'}
NO_WINDOW=getattr(subprocess,'CREATE_NO_WINDOW',0)


@contextmanager
def owned_postgres(out):
    config=json.loads(PRIVATE.read_text('utf-8'));runtime=Path(config['runtime']).resolve()
    no_links(DATA);no_links(runtime)
    if Path(config['pgdata']).resolve()!=DATA.resolve() or not runtime.is_relative_to((ROOT/'.venv').resolve()):
        raise RuntimeError('test_runtime_mismatch')
    pidfile=DATA/'postmaster.pid'
    if pidfile.exists():raise RuntimeError('test_cluster_already_running')
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    config['dsn']=make_conninfo(config['dsn'],host='127.0.0.1',port=port)
    pg_ctl=runtime/'bin/pg_ctl.exe';identity=None
    try:
        with (out/'postgres-control.log').open('a',encoding='utf-8') as log:
            try:
                result=subprocess.run([str(pg_ctl),'-D',str(DATA),'-l',str(out/'postgres-server.log'),
                    '-o',f'-p {port} -h 127.0.0.1','-w','-t','30','start'],stdin=subprocess.DEVNULL,
                    stdout=log,stderr=log,creationflags=NO_WINDOW,timeout=45)
            finally:
                if pidfile.exists():
                    lines=pidfile.read_text('utf-8').splitlines()
                    if Path(lines[1]).resolve()!=DATA.resolve() or int(lines[3])!=port:
                        raise RuntimeError('test_cluster_identity_mismatch')
                    identity=lines[:4]
            if result.returncode:raise RuntimeError('test_cluster_start_failed')
        yield config
    finally:
        if identity and pidfile.exists():
            if pidfile.read_text('utf-8').splitlines()[:4]!=identity:raise RuntimeError('test_cluster_identity_changed')
            with (out/'postgres-control.log').open('a',encoding='utf-8') as log:
                result=subprocess.run([str(pg_ctl),'-D',str(DATA),'-w','-t','30','-m','fast','stop'],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=NO_WINDOW,timeout=45)
            if result.returncode or pidfile.exists():raise RuntimeError('test_cluster_stop_failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-m01-schema-and-synthetic-tests',action='store_true')
    parser.add_argument('--apply-reviewed-schema',action='store_true')
    parser.add_argument('--verify-persistent',action='store_true')
    parser.add_argument('--verify-web',action='store_true')
    parser.add_argument('--verify-legacy-files',action='store_true')
    args=parser.parse_args()
    if not args.approved_m01_schema_and_synthetic_tests:parser.error('Specific M01 005 and synthetic-file scope approval required')
    from m01_database import sql_for,apply_postgres,apply_sqlite
    for kind,digest in HASHES.items():
        if hashlib.sha256(sql_for(kind).encode()).hexdigest()!=digest:parser.error('Reviewed schema hash changed')
    out=ROOT/'output/validation/m01'/('persistent-'+uuid4().hex);out.mkdir(parents=True)
    report={'task':'M01-F2','schema_applied':[]}
    try:
        if args.apply_reviewed_schema:
            report['schema_applied'].append(apply_sqlite(sql_for('sqlite')))
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        with owned_postgres(out) as config:
            if args.apply_reviewed_schema:report['schema_applied'].append(apply_postgres(sql_for('postgres'),config))
            if args.verify_persistent:
                result=subprocess.run([str(ROOT/'.venv/m01-ui/Scripts/python.exe'),'-X','utf8',str(ROOT/'scripts/verify_m01_persistent.py'),
                    '--approved-m01-synthetic-tests','--output',str(out)],input=json.dumps(config)+'\n',text=True,encoding='utf-8',
                    cwd=ROOT,capture_output=True,creationflags=NO_WINDOW,timeout=600)
                (out/'test-output.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
                if result.returncode:raise RuntimeError('persistent_verification_failed')
                report['persistent_verified']=True
            for enabled,script,arguments,name in ((args.verify_web,'verify_m01_web.py',[str(out)],'web'),
                    (args.verify_legacy_files,'verify_p07_jobs.py',['--approved-p07-schema-and-test-files'],'legacy-files')):
                if enabled:
                    result=subprocess.run([str(ROOT/'.venv/m01-ui/Scripts/python.exe'),'-X','utf8',str(ROOT/'scripts'/script),*arguments],
                        input=json.dumps(config)+'\n',text=True,encoding='utf-8',cwd=ROOT,capture_output=True,creationflags=NO_WINDOW,timeout=420 if name=='web' else 240)
                    (out/(name+'-output.txt')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
                    if result.returncode:raise RuntimeError(name+'_verification_failed')
                    report[name+'_verified']=True
        report['postgres_stopped']=True
    except Exception as exc:
        report['error_type']=type(exc).__name__
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'report':str(out.relative_to(ROOT)),'error_type':type(exc).__name__}))
        raise SystemExit(1) from None
    report['schema_hashes']=HASHES
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str((out/'report.json').relative_to(ROOT)),**report}))


if __name__=='__main__':main()
