"""Explicit real-store P06 checks after schema approval; preserves rows, no DROP/DELETE."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import queue
import secrets
import subprocess
import sys
import threading
import time
from uuid import uuid4
from fastapi.testclient import TestClient
from phonetic_core import __version__
from ptb_api.account_store import PostgresAccountStore
from ptb_api.auth import AuthSettings
from ptb_api.job_models import JobInput
from ptb_api.main import create_app
from ptb_worker.store import JobError, PostgresJobStore, SQLiteJobStore, LOCAL_PROJECT

ROOT=Path(__file__).resolve().parents[1]


def wait_for(check, timeout=15):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        result=check()
        if result:return result
        time.sleep(0.1)
    raise AssertionError('Timed out waiting for P06 state')


def worker(config, delay=0):
    child=subprocess.Popen([sys.executable,'-m','ptb_worker.cli','--probe-step-delay',str(delay)],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    child.stdin.write(json.dumps(config)+'\n');child.stdin.flush()
    ready=queue.Queue()
    threading.Thread(target=lambda:ready.put(child.stdout.readline()),daemon=True).start()
    try:assert json.loads(ready.get(timeout=10))['ready'] is True
    except Exception:
        child.terminate();child.wait(timeout=5);child.stdin.close();child.stdout.close();raise
    return child


def stop_worker(child):
    child.stdin.close()
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=5)
    child.stdout.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-test-data',action='store_true')
    args=parser.parse_args()
    if not args.approved_test_data:parser.error('Explicit approval for isolated P06 test data required')
    config=json.loads(sys.stdin.readline());kind=config['kind']
    store=PostgresJobStore(config['dsn'],lease_seconds=2) if kind=='postgres' else SQLiteJobStore(config['path'],lease_seconds=2)
    config['lease_seconds']=2;store.check_schema()
    suffix=uuid4().hex[:10];people=[];created=[];children=[];checks={}
    if kind=='postgres':
        accounts=PostgresAccountStore(config['dsn'])
        with accounts.connection() as conn:
            assert conn.execute('SELECT current_database() AS name').fetchone()['name']=='ptb_p05_test_20260909'
        for index in range(10):
            password=secrets.token_urlsafe(32);username=f'p06_{suffix}_{index}'
            user=accounts.create_user(username,password)
            project=accounts.create_project(str(user['id']),'P06 任务验证')
            people.append((str(user['id']),str(project['id']),username,password))
    else:
        assert store.path==ROOT/'output/validation/p06/local-state.sqlite3'
        people=[('local',LOCAL_PROJECT,None,None)]
    owner,project,*_=people[0]

    def submit(person=people[0],key=None):
        body=JobInput(project_id=person[1],idempotency_key=key or uuid4().hex)
        result=store.submit(person[0],body);created.append((person[0],result['id']))
        return result,body

    def cancel_all():
        for account,job in created:store.cancel(account,job)
        # Owned fake claims may be cancel_requested. Expiration tests handle real worker cases separately.

    def manifest():
        return dict(complete=True,kind='pipeline_check_metadata',sha256=hashlib.sha256(bytes(range(256))*16).hexdigest(),sample_count=4096,core_version=__version__)

    try:
        job,body=submit();assert store.submit(owner,body)['id']==job['id']
        with ThreadPoolExecutor(max_workers=8) as pool:
            duplicates=list(pool.map(lambda _:store.submit(owner,body)['id'],range(8)))
        assert set(duplicates)=={job['id']}
        try:store.submit(owner,body.model_copy(update={'config':body.config.model_copy(update={'seed':1})}))
        except JobError as e:assert e.code=='idempotency_conflict'
        else:raise AssertionError('Different payload reused idempotency key')
        checks['concurrent_idempotency_and_payload_conflict']=True
        assert store.cancel(owner,job['id'])['state']=='cancelled'
        jobs=[submit(person)[0] for person in people]
        extra,_=submit()
        with ThreadPoolExecutor(max_workers=8) as pool:
            claims=[c for c in pool.map(lambda _:store.claim(str(uuid4())),range(8)) if c]
        assert len(claims)==min(2,len(people))
        assert len({str(c['owner_id']) for c in claims})==len(claims)
        assert store.claim('extra-worker') is None
        checks['global_two_and_per_owner_one']=True
        if kind=='postgres':checks['ten_accounts_can_queue']=True
        for claim in claims:
            identity=(claim['id'],claim['worker_id'],claim['generation'])
            assert not store.finish(claim['id'],claim['worker_id'],claim['generation']-1,result=manifest())
            assert store.finish(*identity,result=manifest())
            assert store.get(str(claim['owner_id']),claim['id'])['result_manifest']['complete']
        cancel_all()
        checks['old_generation_rejected_and_atomic_manifest']=True

        job,_=submit();claim=store.claim('lease-test');assert claim['id']==job['id']
        with store.transaction() as tx:
            tx.execute('UPDATE {jobs} SET lease_until=? WHERE id=?',(tx.now()-1,job['id']))
        assert store.get(owner,job['id'])['state']=='interrupted'
        assert not store.finish(job['id'],'lease-test',claim['generation'],result=manifest())
        retry=store.retry(owner,job['id'],uuid4().hex);created.append((owner,retry['id']))
        assert retry['id']!=job['id'] and retry['retry_of']==job['id']
        store.cancel(owner,retry['id']);checks['expired_lease_fenced_and_explicit_retry']=True

        job,_=submit();claim=store.claim('race')
        barrier=threading.Barrier(2)
        def cancel():barrier.wait();return store.cancel(owner,job['id'])
        def finish():barrier.wait();return store.finish(job['id'],'race',claim['generation'],result=manifest())
        with ThreadPoolExecutor(max_workers=2) as pool:
            a,b=pool.submit(cancel),pool.submit(finish);a.result();b.result()
        final=store.get(owner,job['id']);assert final['state'] in ('cancelled','succeeded')
        assert (final['state']=='succeeded') == (final['result_manifest'] is not None)
        checks['cancel_finish_race_serialized']=True
        events=store.events(owner,job['id'],0)
        assert [e['sequence'] for e in events]==list(range(1,len(events)+1))
        assert store.events(owner,job['id'],events[-1]['sequence'])==[]
        checks['event_sequence_and_resume_cursor']=True

        job,_=submit()
        try:
            with store.transaction() as tx:
                tx.execute('UPDATE {jobs} SET progress=0.5 WHERE id=?',(job['id'],))
                raise RuntimeError('rollback probe')
        except RuntimeError:pass
        assert store.get(owner,job['id'])['progress']==0
        rebuilt=PostgresJobStore(config['dsn']) if kind=='postgres' else SQLiteJobStore(config['path'])
        assert rebuilt.get(owner,job['id'])['state']=='queued'
        store.cancel(owner,job['id']);checks['transaction_rollback_and_adapter_reopen']=True

        if kind=='postgres':
            settings=AuthSettings(origin='https://p06.test',signing_key=secrets.token_urlsafe(48))
            app=create_app(account_store=accounts,auth_settings=settings,job_store=store)
            def login(client,person):
                challenge=client.get('/api/v1/auth/challenge').json()['csrf_token']
                r=client.post('/api/v1/auth/login',json={'username':person[2],'password':person[3]},headers={'Origin':settings.origin,'X-CSRF-Token':challenge})
                assert r.status_code==200
                return {'Origin':settings.origin,'X-CSRF-Token':r.json()['csrf_token'],'X-PTB-Account':person[0]}
            # Give this test run its own simulated peer, so its login budget does
            # not mutate P05's earlier default 'testclient' rate-limit bucket.
            with TestClient(app,base_url=settings.origin,client=('p06-'+suffix,50000)) as a,TestClient(app,base_url=settings.origin,client=('p06-'+suffix,50001)) as b:
                ah,bh=login(a,people[0]),login(b,people[1]);job,_=submit()
                for path in ('','/events'):
                    assert b.get('/api/v1/jobs/'+job['id']+path,headers=bh).status_code==404
                assert b.post('/api/v1/jobs/'+job['id']+'/cancel',headers=bh).status_code==404
                assert b.post('/api/v1/jobs/'+job['id']+'/retry',json={'idempotency_key':uuid4().hex},headers=bh).status_code==404
                assert b.post('/api/v1/jobs',json={'project_id':project,'idempotency_key':uuid4().hex},headers=bh).status_code==404
                assert a.get('/api/v1/jobs/'+job['id'],headers=ah).status_code==200
                assert a.post('/api/v1/jobs/'+job['id']+'/cancel',headers={'Origin':settings.origin}).status_code==403
                store.cancel(owner,job['id'])
            checks['real_accounts_job_events_cancel_retry_owner_isolation']=True

        job,_=submit();child=worker(config,0.05);children.append(child)
        final=wait_for(lambda:(r if (r:=store.get(owner,job['id']))['state']=='succeeded' else None))
        assert final['result_manifest']['sha256']==manifest()['sha256']
        stop_worker(child);children.remove(child)
        checks['real_worker_core_subprocess_result']=True

        job,_=submit();child=worker(config,0.2);children.append(child)
        wait_for(lambda:store.get(owner,job['id'])['state']=='running')
        store.cancel(owner,job['id'])
        wait_for(lambda:store.get(owner,job['id'])['state']=='cancelled')
        stop_worker(child);children.remove(child)
        checks['running_worker_cancellation']=True

        job,_=submit();child=worker(config,0.2);children.append(child)
        wait_for(lambda:store.get(owner,job['id'])['state']=='running')
        child.terminate();child.wait(timeout=5);child.stdin.close();child.stdout.close();children.remove(child)
        wait_for(lambda:store.get(owner,job['id'])['state']=='interrupted',timeout=8)
        retry=store.retry(owner,job['id'],uuid4().hex);created.append((owner,retry['id']))
        child=worker(config);children.append(child)
        wait_for(lambda:store.get(owner,retry['id'])['state']=='succeeded')
        stop_worker(child);children.remove(child)
        checks['killed_worker_interrupted_then_retry_with_new_worker']=True
    finally:
        for child in children:stop_worker(child)
        cancel_all()
    output=ROOT/'output/validation/p06';output.mkdir(parents=True,exist_ok=True)
    report={'kind':kind,'checks':checks,'scope':'P06 metadata probe only; no file quota, scientific algorithm or native-device claim'}
    (output/f'{kind}-{suffix}.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
