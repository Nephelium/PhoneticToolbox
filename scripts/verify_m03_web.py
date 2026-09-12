"""Owned PostgreSQL + API/worker + independent Chrome. No schema mutations."""
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
from ptb_worker.store import PostgresJobStore
from run_m01_validation import owned_postgres
import hashlib
import numpy as np
from scipy.io import wavfile

ROOT=Path(__file__).resolve().parents[1]


def verify(config,out):
    account=PostgresAccountStore(config['dsn']);storage=Storage(config['dsn'],ROOT/'output/validation/p07/storage')
    jobs=PostgresJobStore(config['dsn']);storage.recover();people=[]
    with jobs.transaction(write=False) as tx:
        assert not tx.execute("SELECT 1 FROM {jobs} WHERE state IN ('queued','running','cancel_requested')").fetchone(),'Existing active jobs'
    for i in range(2):
        name='m03e2_'+uuid4().hex[:12];password=secrets.token_urlsafe(32)
        user=account.create_user(name,password);project=account.create_project(str(user['id']),'EGG 网页验证')
        people.append(dict(username=name,password=password,id=str(user['id']),project=str(project['id'])))
    inputs=out/'inputs';inputs.mkdir()
    with np.load(ROOT/'tests/fixtures/m03/EGG-SYN-PCM16.npz') as a:samples=np.column_stack([a['load.egg_signal_raw'],a['load.audio_signal']])
    wavfile.write(inputs/'EGG ɑ̃˥.wav',44100,samples)
    wavfile.write(inputs/'silent.wav',44100,np.zeros_like(samples))
    os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    origin=f'http://127.0.0.1:{port}';options=dict(dsn=config['dsn'],storage_root=str(storage.root),enable_acoustic_batches=True)
    hidden=getattr(subprocess,'CREATE_NO_WINDOW',0);server=worker=None
    try:
        server=subprocess.Popen([sys.executable,'-m','ptb_api.server','--port',str(port),'--frontend',str(ROOT/'frontend/dist'),'--managed'],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',creationflags=hidden)
        server.stdin.write(json.dumps(options|dict(signing_key=secrets.token_urlsafe(48),enable_jobs=True,enable_file_jobs=True))+'\n');server.stdin.flush()
        deadline=time.monotonic()+20;opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            try:
                with opener.open(origin+'/api/v1/health',timeout=1) as response:assert json.load(response)['mode']=='server'
                break
            except OSError:
                if time.monotonic()>deadline or server.poll() is not None:raise RuntimeError('Owned API unavailable')
                time.sleep(.1)
        worker=subprocess.Popen([sys.executable,'-m','ptb_worker.cli'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',creationflags=hidden)
        worker.stdin.write(json.dumps(options|dict(kind='postgres',reaper_binary=str(ROOT/'phonetic_toolbox/core/acoustic/reaper.exe')))+'\n');worker.stdin.flush();assert json.loads(worker.stdout.readline())=={'ready':True}
        runtime=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
        settings=dict(origin=origin,people=people,playwright=str(runtime/'node_modules/playwright'),browser='C:/Program Files/Google/Chrome/Application/chrome.exe',output=str(out),uploads=[str(inputs/n) for n in ('EGG ɑ̃˥.wav','silent.wav')])
        done=subprocess.run([str(runtime/'bin/node.exe'),str(ROOT/'tests/e2e/m03-web.cjs')],input=json.dumps(settings)+'\n',capture_output=True,text=True,encoding='utf-8',creationflags=hidden,timeout=360)
        log=done.stdout+'\n'+done.stderr
        for person in people:log=log.replace(person['password'],'[redacted]')
        (out/'browser.log').write_text(log,encoding='utf-8');assert done.returncode==0,'See browser.log'
        import soundfile as sf
        downloaded=json.loads((out/'downloads.json').read_text('utf-8'))
        for item in downloaded:
            target=out/'downloads'/item['saved']
            assert hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']
            if item['name'].endswith('.wav'):
                audio,sr=sf.read(target);assert sr==44100 and len(audio)==5292
        report=json.loads((out/'web-report.json').read_text('utf-8'))
        for identifier in report['job_ids']:assert jobs.get(people[0]['id'],identifier)['state']=='succeeded'
        with storage._locked() as conn:
            for person in people:
                assert not conn.execute("SELECT 1 FROM ptb_storage.assets WHERE owner_id=%s AND kind='temporary' AND state!='deleted'",(person['id'],)).fetchone()
        from ptb_api.storage_models import UploadInput
        from ptb_api.quota import QUOTA_BYTES
        used=storage.usage(people[0]['id'])
        reservation=storage.create(people[0]['id'],UploadInput(project_id=people[0]['project'],name='owned-quota-reservation.bin',expected_bytes=QUOTA_BYTES-used['used_bytes']-used['reserved_bytes'],idempotency_key=uuid4().hex))
        try:
            expired=downloaded[0]['id']
            with storage._locked() as conn:
                now=storage._now(conn)
                assert conn.execute('SELECT owner_id FROM ptb_storage.assets WHERE id=%s',(expired,)).fetchone()['owner_id']==__import__('uuid').UUID(people[0]['id'])
                conn.execute('UPDATE ptb_storage.assets SET expires_at=%s WHERE id=%s AND owner_id=%s',(now-1,expired,people[0]['id']))
                conn.execute("UPDATE ptb_storage.assets SET expires_at=%s WHERE owner_id=%s AND project_id=%s AND name='silent.wav'",(now+240,people[0]['id'],people[0]['project']))
            stage=subprocess.run([str(runtime/'bin/node.exe'),str(ROOT/'tests/e2e/m03-web.cjs')],input=json.dumps(settings|dict(stage='limits',expired=expired))+'\n',capture_output=True,text=True,encoding='utf-8',creationflags=hidden,timeout=150)
            log=stage.stdout+'\n'+stage.stderr
            for person in people:log=log.replace(person['password'],'[redacted]')
            (out/'limits-browser.log').write_text(log,encoding='utf-8');assert stage.returncode==0,'See limits-browser.log'
        finally:storage.delete(people[0]['id'],reservation['id'])
        # Deadlines above were deliberately moved by this test. Invoke the same
        # cleanup routine explicitly; this is not an elapsed-seven-days test.
        storage.cleanup()
        assert not storage._path(expired).exists()
        with storage._locked() as conn:
            assert not conn.execute("SELECT 1 FROM ptb_storage.assets WHERE owner_id=%s AND kind='temporary' AND state!='deleted'",(people[0]['id'],)).fetchone()
        assert storage.usage(people[0]['id'])['reserved_bytes']==0
        return dict(browser_verified=True,output_readback=True,no_live_temporary_files=True,quota_and_expiry_verified=True,controlled_expiry_cleanup_removed_file=True)


    finally:
        for process in (worker,server):
            if process:
                if process.stdin:process.stdin.close()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5)


def main():
    out=ROOT/'output/validation/m03-e2'/('web-'+uuid4().hex);out.mkdir(parents=True);report={'schema_applied':[],'success':False}
    try:
        with owned_postgres(out) as config:report.update(verify(config,out))
        report.update(success=True,postgres_stopped=True)
    finally:
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(out/'report.json')


if __name__=='__main__':main()
