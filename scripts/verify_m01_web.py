"""Owned actual PostgreSQL API/worker + independent Chrome, synthetic records only."""
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from uuid import uuid4
import urllib.request
from ptb_api.account_store import PostgresAccountStore
from ptb_api.storage import Storage
from ptb_api.storage_models import UploadInput
from ptb_worker.store import PostgresJobStore
from ptb_worker.acoustic_files import AcousticFiles
from ptb_worker.acoustic_batches import AcousticBatches
from baseline_support import RECIPES,create_fixture

ROOT=Path(__file__).resolve().parents[1]


def verify(config,out):
    account=PostgresAccountStore(config['dsn']);storage=Storage(config['dsn'],ROOT/'output/validation/p07/storage')
    jobs=PostgresJobStore(config['dsn']);files=AcousticFiles(jobs,storage);batches=AcousticBatches(jobs,files)
    storage.recover()
    people=[];server=worker=None
    for i in range(2):
        name='m01web_'+uuid4().hex[:12];password=secrets.token_urlsafe(32)
        user=account.create_user(name,password);project=account.create_project(str(user['id']),'批次验证项目')
        people.append(dict(username=name,password=password,id=str(user['id']),project=str(project['id'])))
    source=create_fixture(out,RECIPES[0]);raw=source.read_bytes()
    for name,data in [(source.name,raw),(source.stem+'.TextGrid','File type = "ooTextFile short"\n"TextGrid"\n0\n.8\n<exists>\n1\n"IntervalTier"\n"音节"\n0\n.8\n2\n0\n.4\n"阴平"\n.4\n.8\n"上声"\n'.encode())]:
        a=storage.create(people[0]['id'],UploadInput(project_id=people[0]['project'],name=name,expected_bytes=len(data),idempotency_key=uuid4().hex))
        for offset in range(0,len(data),65536):storage.append(people[0]['id'],a['id'],offset,data[offset:offset+65536])
        storage.finalize(people[0]['id'],a['id'])
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    origin=f'http://127.0.0.1:{port}';options=dict(dsn=config['dsn'],storage_root=str(storage.root),enable_acoustic_batches=True)
    hidden=getattr(subprocess,'CREATE_NO_WINDOW',0)
    try:
        server=subprocess.Popen([sys.executable,'-m','ptb_api.server','--port',str(port),'--frontend',str(ROOT/'frontend/dist'),'--managed'],
            stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',creationflags=hidden)
        server.stdin.write(json.dumps(options|dict(signing_key=secrets.token_urlsafe(48),enable_jobs=True,enable_file_jobs=True))+'\n');server.stdin.flush()
        deadline=time.monotonic()+15;opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            try:
                with opener.open(origin+'/api/v1/health',timeout=1) as r:assert json.load(r)['mode']=='server'
                break
            except OSError:
                if time.monotonic()>deadline or server.poll() is not None:raise RuntimeError('Owned test API unavailable')
                time.sleep(.1)
        worker=subprocess.Popen([sys.executable,'-m','ptb_worker.cli'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',creationflags=hidden)
        worker.stdin.write(json.dumps(options|dict(kind='postgres',reaper_binary=str(ROOT/'phonetic_toolbox/core/acoustic/reaper.exe')))+'\n');worker.stdin.flush()
        assert json.loads(worker.stdout.readline())=={'ready':True}
        runtime=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
        browser_input=dict(origin=origin,people=people,playwright=str(runtime/'node_modules/playwright'),browser='C:/Program Files/Google/Chrome/Application/chrome.exe',output=str(out))
        done=subprocess.run([str(runtime/'bin/node.exe'),str(ROOT/'tests/e2e/m01-tasks.cjs')],input=json.dumps(browser_input)+'\n',capture_output=True,text=True,encoding='utf-8',creationflags=hidden,timeout=160)
        # Test assertions never log auth/session inputs; redact even unexpected failures.
        log=done.stdout+'\n'+done.stderr
        for person in people:log=log.replace(person['password'],'[redacted]')
        (out/'browser.log').write_text(log,encoding='utf-8');assert done.returncode==0,'See browser.log'
        ttl=[]
        for job in jobs.list(people[0]['id'],people[0]['project']):
            assert job['state']=='succeeded'
            with jobs.transaction(write=False) as tx:stored=jobs._row(tx,job['id'])
            inputs=json.loads(stored['snapshot'])['input_assets'];deadline=min(a['expires_at'] for a in inputs)
            expiries={f['expires_at'] for f in job['result_manifest']['files']};assert len(expiries)==1
            due=expiries.pop()
            if job['operation']=='acoustic_analysis':assert deadline<due<=job['updated_at']+604800
            else:assert due<=deadline
            ttl.append(dict(operation=job['operation'],input_expiry=deadline,output_expiry=due))
        (out/'web-ttl.json').write_text(json.dumps(ttl,indent=2),encoding='utf-8')
    finally:
        for person in people:
            for batch in batches.list(person['id'],person['project']):batches.cancel(person['id'],batch['id'])
        for process in (worker,server):
            if process:
                process.stdin.close()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5)
                if process.stdout:process.stdout.close()
        (out/'web-processes.json').write_text(json.dumps(dict(server_exit=server.returncode if server else None,worker_exit=worker.returncode if worker else None,people=[dict(id=p['id'],project=p['project']) for p in people]),indent=2),encoding='utf-8')
    assert server.returncode==worker.returncode==0


if __name__=='__main__':verify(json.loads(sys.stdin.readline()),Path(sys.argv[1]))
