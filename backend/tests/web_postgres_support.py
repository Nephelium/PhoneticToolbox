"""Owned synthetic PostgreSQL with retained diagnostics and scratch cleanup."""
from pathlib import Path
import importlib.util
import os
import socket
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('web_verification_artifacts',ROOT/'scripts/verification_artifacts.py')
artifacts=importlib.util.module_from_spec(spec);spec.loader.exec_module(artifacts)


@pytest.fixture(scope='module')
def cluster():
    if os.environ.get('PTB_POLICY_FRESH_PG')!='1':pytest.skip('explicit isolated PostgreSQL opt-in required')
    binaries=ROOT/'.venv/postgresql-17.11-3/pgsql/bin'
    if os.name!='nt' or not (binaries/'initdb.exe').exists():pytest.skip('bundled Windows PostgreSQL required')
    with artifacts.verification_run('web-launch','postgres','Synthetic users, fresh initdb, migrations 001-006/008-009; no external SMTP or existing DB') as (out,scratch):
        data=scratch/'pgdata'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        def run(args):
            with (out/'commands.log').open('a',encoding='utf-8') as log:
                result=subprocess.run([str(binaries/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW,timeout=60)
            assert result.returncode==0,'See isolated PostgreSQL commands.log'
        run(['initdb.exe','-D',str(data),'-U','policy_test','-A','trust','--no-locale','-E','UTF8'])
        try:
            run(['pg_ctl.exe','-D',str(data),'-l',str(out/'postgres.log'),'-w','start',
                '-o',f'-h 127.0.0.1 -p {port} -c max_connections=20 -c shared_buffers=16MB'])
            yield dict(host='127.0.0.1',port=port,user='policy_test',dbname='postgres'),scratch
        finally:
            if (data/'postmaster.pid').exists():run(['pg_ctl.exe','-D',str(data),'-w','stop','-m','fast'])
            assert not (data/'postmaster.pid').exists()
