"""M12 existing dedicated PostgreSQL + real API and isolated Chrome. No DDL."""
import hashlib
import json
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
from uuid import uuid4
from ptb_api.account_store import PostgresAccountStore
from ptb_api.storage import Storage
from ptb_api.storage_models import UploadInput
from ptb_api.quota import QUOTA_BYTES
from ptb_worker.io.annotation import lip_preview
from run_m01_validation import owned_postgres
from m12_ui_bridge import fixtures

ROOT=Path(__file__).resolve().parents[1]


def verify(config,out):
    account=PostgresAccountStore(config['dsn']);account.check_schema()
    storage=Storage(config['dsn'],ROOT/'output/validation/p07/storage');storage.recover();people=[]
    for _ in range(2):
        username='m12_'+uuid4().hex[:12];password=secrets.token_urlsafe(32)
        user=account.create_user(username,password);project=account.create_project(str(user['id']),'M12 网页验证')
        people.append(dict(username=username,password=password,id=str(user['id']),project=str(project['id'])))
    inputs=fixtures(out)
    wire=lip_preview((inputs/'audio_recording.pkl').read_bytes(),'audio_recording.pkl')
    (inputs/'audio_recording.lip.json').write_text(json.dumps(wire,ensure_ascii=False),encoding='utf-8')
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    origin=f'http://127.0.0.1:{port}';hidden=getattr(subprocess,'CREATE_NO_WINDOW',0);server=None
    try:
        server=subprocess.Popen([sys.executable,'-m','ptb_api.server','--port',str(port),'--frontend',str(ROOT/'frontend/dist'),'--managed'],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',creationflags=hidden)
        server.stdin.write(json.dumps(dict(dsn=config['dsn'],storage_root=str(storage.root),signing_key=secrets.token_urlsafe(48)))+'\n');server.stdin.flush()
        deadline=time.monotonic()+20;opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            try:
                with opener.open(origin+'/api/v1/health',timeout=1) as response:assert json.load(response)['mode']=='server'
                break
            except OSError:
                if time.monotonic()>deadline or server.poll() is not None:raise RuntimeError('Owned API unavailable')
                time.sleep(.1)
        runtime=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
        settings=dict(origin=origin,people=people,playwright=str(runtime/'node_modules/playwright'),browser='C:/Program Files/Google/Chrome/Application/chrome.exe',output=str(out),uploads=[str(inputs/('audio_recording'+ext)) for ext in ('.wav','.TextGrid','.lab','.lip.json')])
        def browser(stage='main',**extra):
            done=subprocess.run([str(runtime/'bin/node.exe'),str(ROOT/'tests/e2e/m12-web.cjs')],input=json.dumps(settings|dict(stage=stage)|extra)+'\n',capture_output=True,text=True,encoding='utf-8',creationflags=hidden,timeout=160)
            log=done.stdout+'\n'+done.stderr
            for person in people:log=log.replace(person['password'],'[redacted]')
            (out/(stage+'-browser.log')).write_text(log,encoding='utf-8')
            assert done.returncode==0,'See '+stage+'-browser.log'
        browser()
        report=json.loads((out/'web-report.json').read_text('utf-8'));expired=report['original_id']
        with storage._locked() as conn:
            conn.execute('UPDATE ptb_storage.assets SET expires_at=0 WHERE id=%s AND owner_id=%s',(expired,people[0]['id']))
        usage=storage.usage(people[0]['id'])
        reservation=storage.create(people[0]['id'],UploadInput(project_id=people[0]['project'],name='m12-quota-test.bin',expected_bytes=QUOTA_BYTES-usage['used_bytes']-usage['reserved_bytes'],idempotency_key=uuid4().hex))
        try:browser('limits',expired=expired)
        finally:storage.delete(people[0]['id'],reservation['id'])
        storage.cleanup()
        assert not storage._path(expired).exists()
        assert storage.usage(people[0]['id'])['reserved_bytes']==0
        return dict(success=True,actual_pg_accounts=2,browser=True,quota=True,controlled_expiry=True,expired_physically_removed=True,schemas_applied=[])
    finally:
        if server:
            server.stdin.close()
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:server.terminate();server.wait(timeout=5)


def main():
    out=ROOT/'output/validation/m12-web'/uuid4().hex;out.mkdir(parents=True);report={'success':False}
    try:
        with owned_postgres(out) as config:report.update(verify(config,out))
        report['owned_postgres_stopped']=True
    finally:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(out/'report.json')


if __name__=='__main__':main()
