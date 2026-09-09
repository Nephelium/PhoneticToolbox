"""Approved isolated PostgreSQL UI acceptance host. Random credentials stay in a private file."""
import argparse
import json
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from uuid import uuid4
import httpx
from psycopg.conninfo import make_conninfo
from ptb_api.account_store import PostgresAccountStore
from ptb_worker.store import PostgresJobStore

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/validation/p06'
PRIVATE=ROOT/'output/validation/p05/postgres-private'
DATA=ROOT/'output/validation/p05/postgres-data'
NO_WINDOW=getattr(subprocess,'CREATE_NO_WINDOW',0)


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-test-data',action='store_true')
    if not parser.parse_args().approved_test_data:parser.error('Requires approved P06 test account and task data')
    config=json.loads((PRIVATE/'connection.json').read_text('utf-8'))
    runtime=Path(config['runtime']).resolve()
    assert Path(config['pgdata']).resolve()==DATA.resolve() and runtime.is_relative_to((ROOT/'.venv').resolve())
    assert not (DATA/'postmaster.pid').exists()
    pgport,apiport=port(),port()
    config['dsn']=make_conninfo(config['dsn'],port=pgport)
    config['enable_jobs']=True
    suffix=uuid4().hex[:10]
    signal=OUT/f'ui-stop-{suffix}.signal'
    children=[];started=False
    pg_ctl=runtime/'bin/pg_ctl.exe'
    try:
        with (OUT/'ui-pg-control.log').open('a',encoding='utf-8') as log:
            try:
                p=subprocess.run([str(pg_ctl),'-D',str(DATA),'-l',str(OUT/'ui-pg-server.log'),'-o',f'-p {pgport} -h 127.0.0.1','-w','-t','30','start'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=NO_WINDOW,timeout=45)
            finally:started=(DATA/'postmaster.pid').exists()
        assert p.returncode==0
        jobs=PostgresJobStore(config['dsn']);jobs.check_schema()
        accounts=PostgresAccountStore(config['dsn'])
        people=[]
        for letter in ('a','b'):
            username=f'ui_{letter}_{suffix}';password=secrets.token_urlsafe(32)
            accounts.create_user(username,password);people.append({'username':username,'password':password})
        (PRIVATE/'p06-ui-credentials.json').write_text(json.dumps({'accounts':people}),encoding='utf-8')
        with (OUT/'ui-process.log').open('a',encoding='utf-8') as log:
            api=subprocess.Popen([sys.executable,'-m','ptb_api.server','--port',str(apiport),'--frontend',str(ROOT/'frontend/dist'),'--managed'],
                                 stdin=subprocess.PIPE,stdout=log,stderr=log,text=True,encoding='utf-8',creationflags=NO_WINDOW)
            children.append(api);api.stdin.write(json.dumps(config)+'\n');api.stdin.flush()
            worker=subprocess.Popen([sys.executable,'-m','ptb_worker.cli','--probe-step-delay','0.15'],
                                    stdin=subprocess.PIPE,stdout=log,stderr=log,text=True,encoding='utf-8',creationflags=NO_WINDOW)
            children.append(worker);worker.stdin.write(json.dumps({'kind':'postgres','dsn':config['dsn']})+'\n');worker.stdin.flush()
            url=f'http://127.0.0.1:{apiport}'
            with httpx.Client(base_url=url,trust_env=False,timeout=3) as client:
                for _ in range(100):
                    assert all(p.poll() is None for p in children)
                    try:
                        if client.get('/api/v1/health').status_code==200:break
                    except httpx.TransportError:pass
                    time.sleep(0.1)
                else:raise RuntimeError('UI API readiness timed out')
            (OUT/'ui-host.json').write_text(json.dumps({'url':url+'/server/','stop_signal':str(signal),'owned_pids':[p.pid for p in children]}),encoding='utf-8')
            print(json.dumps({'url':url+'/server/','stop_signal':str(signal)}),flush=True)
            while not signal.exists():
                assert all(p.poll() is None for p in children)
                time.sleep(0.2)
    finally:
        for child in reversed(children):
            child.stdin.close()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=5)
        if started:
            lines=(DATA/'postmaster.pid').read_text('utf-8').splitlines()
            assert Path(lines[1]).resolve()==DATA.resolve()
            subprocess.run([str(pg_ctl),'-D',str(DATA),'-w','-t','30','-m','fast','stop'],check=True,creationflags=NO_WINDOW)
        (OUT/'ui-shutdown.json').write_text(json.dumps({'child_exit_codes':[p.returncode for p in children],'postgres_stopped':not (DATA/'postmaster.pid').exists()}),encoding='utf-8')


if __name__=='__main__':main()
