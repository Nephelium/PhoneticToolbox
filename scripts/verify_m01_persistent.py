"""Real approved M01 synthetic PostgreSQL tasks, publication and recovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import secrets
import sys
import threading
import time
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
from ptb_api.account_store import PostgresAccountStore
from ptb_api.storage import Storage
from ptb_api.storage_models import UploadInput
from ptb_api.acoustic_batch_models import BatchRequest,AcousticTaskManifest
from ptb_api.job_models import JobInput
from ptb_worker.store import PostgresJobStore,JobError
from ptb_worker.acoustic_files import AcousticFiles
from ptb_worker.files import FilePipeline
from ptb_worker.acoustic_batches import AcousticBatches
from ptb_worker.acoustic_executor import execute_acoustic_claim
from m01_database import TABLES,fingerprint

ROOT=Path(__file__).resolve().parents[1]


def verify(config,out):
    jobs=PostgresJobStore(config['dsn'],lease_seconds=2)
    storage=Storage(config['dsn'],ROOT/'output/validation/p07/storage')
    files=AcousticFiles(jobs,storage);files.reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe'
    batches=AcousticBatches(jobs,files)
    with storage._locked() as conn:
        assert not conn.execute("SELECT 1 FROM ptb_jobs.jobs WHERE state IN ('queued','running','cancel_requested')").fetchone(),'Existing active jobs'
        before={t:[dict(r) for r in conn.execute('SELECT * FROM '+t)] for t in TABLES}
    storage.recover()
    account=PostgresAccountStore(config['dsn']);suffix=uuid4().hex[:10];people=[]
    for i in range(2):
        user=account.create_user('m01_'+suffix+str(i),secrets.token_urlsafe(32));owner=str(user['id'])
        project=str(account.create_project(owner,'M01 synthetic verification')['id']);people.append((owner,project))
    owner,project=people[0];assets=[];checks={};owned_batches=[]
    def track(batch):
        owned_batches.append(batch['id'])
        (out/'owned-test-records.json').write_text(json.dumps(dict(people=people,batches=owned_batches),indent=2)+'\n',encoding='utf-8')
        return batch
    def upload(raw,name,person=None):
        who,where=person or people[0]
        a=storage.create(who,UploadInput(project_id=where,name=name,expected_bytes=len(raw),idempotency_key=uuid4().hex))
        for offset in range(0,len(raw),65536):storage.append(who,a['id'],offset,raw[offset:offset+65536])
        a=storage.finalize(who,a['id']);assets.append((who,a['id']));return dict(asset_id=a['id'],sha256=a['sha256'])
    signal=np.column_stack((np.arange(1000,dtype=np.int16),-np.arange(1000,dtype=np.int16)))
    buf=io.BytesIO();wavfile.write(buf,1000,signal);wav=buf.getvalue()
    grid=b'File type = "ooTextFile short"\n"TextGrid"\n0\n1\n<exists>\n1\n"IntervalTier"\n"syllable"\n0\n1\n2\n0\n.2\n"a"\n.2\n1\n"sil"\n'
    tg=upload(grid,'labels.TextGrid')
    refs=[dict(audio=upload(wav,f'file-{i}.wav'),textgrid=tg) for i in range(17)]
    body=BatchRequest(operation='textgrid_segment',project_id=project,idempotency_key=uuid4().hex,inputs=refs,layer='syllable')
    created=track(batches.submit(owner,body))
    assert created['summary']['counts']['queued']==1 and created['summary']['counts']['not_started']==16
    assert created['audio_names']==[f'file-{i}.wav' for i in range(17)]
    other_store=PostgresJobStore(config['dsn'])
    with other_store.transaction(write=False) as tx:assert other_store.capacity_used(tx,owner)==17
    assert batches.submit(owner,body)['id']==created['id']
    body.layer='wrong'
    try:batches.submit(owner,body)
    except JobError as e:assert e.code=='idempotency_conflict'
    else:raise AssertionError('Idempotency mismatch accepted')
    try:batches.get(people[1][0],created['id'])
    except JobError as e:assert e.code=='batch_not_found'
    else:raise AssertionError('Cross-owner batch')
    checks['ordered_17_and_idempotency']=True
    for index in range(17):
        claim=jobs.claim('m01-'+suffix);assert claim and json.loads(claim['snapshot'])['batch_index']==index
        execute_acoustic_claim(jobs,claim,claim['worker_id'],threading.Event())
        result=jobs.get(owner,claim['id']);assert result['state']=='succeeded',result['error_code']
        manifest=AcousticTaskManifest.model_validate(result['result_manifest'])
        item=next(f for f in manifest.files if f.name.endswith('.wav'))
        raw=storage.read_block(owner,item.id,0,item.size_bytes);fs,values=wavfile.read(io.BytesIO(raw))
        assert fs==1000;np.testing.assert_array_equal(values,signal[:200])
        assert item.expires_at<=storage.metadata(owner,tg['asset_id'])['expires_at']
    assert batches.get(owner,created['id'])['summary']['complete']
    checks['17_actual_results']=True
    reopened=AcousticBatches(PostgresJobStore(config['dsn']),files)
    assert reopened.get(owner,created['id'])['summary']['complete']
    # Restore adapter/job pairing after independent read/reconstruction.
    jobs.batches=batches
    checks['reconstructed_batch']=True
    cancel_body=body.model_copy(update={'idempotency_key':uuid4().hex,'layer':'syllable'})
    cancelled=track(batches.submit(owner,cancel_body));cancelled=batches.cancel(owner,cancelled['id'])
    assert cancelled['summary']['closed'] and cancelled['summary']['counts']['not_started']==16
    assert cancelled['summary']['counts']['cancelled']==1 and not cancelled['summary']['complete']
    checks['cancel_queued_keeps_unstarted']=True
    bad=upload(grid.replace(b'syllable',b'other'),'bad.TextGrid')
    failed_body=BatchRequest(operation='textgrid_segment',project_id=project,idempotency_key=uuid4().hex,
                            inputs=[refs[0],dict(audio=refs[1]['audio'],textgrid=bad),refs[2]],layer='syllable')
    failed=track(batches.submit(owner,failed_body))
    for _ in range(3):
        claim=jobs.claim('m01-'+suffix);execute_acoustic_claim(jobs,claim,claim['worker_id'],threading.Event())
    state=batches.get(owner,failed['id'])['summary'];assert state['counts']['succeeded']==2 and state['counts']['failed']==1 and not state['complete']
    checks['middle_failure_keeps_results']=True
    # Concurrent duplicate requests still admit one batch and one first child.
    same=cancel_body.model_copy(update={'idempotency_key':uuid4().hex,'inputs':cancel_body.inputs[:1]})
    with ThreadPoolExecutor(max_workers=2) as pool:
        duplicates=list(pool.map(lambda _:batches.submit(owner,same),range(2)))
    concurrent=track(duplicates[0]);assert duplicates[0]['id']==duplicates[1]['id']
    batches.cancel(owner,concurrent['id']);checks['concurrent_idempotent_admission']=True
    # Cancel after actual native child creation, then confirm its owning handle exits.
    running=track(batches.submit(owner,same.model_copy(update={'idempotency_key':uuid4().hex})))
    claim=jobs.claim('m01-'+suffix);handles=[]
    from ptb_worker.native.windows import open_process,wait,close
    def started(pid):
        handle=open_process(0x100000,False,pid);assert handle;handles.append(handle)
        batches.cancel(owner,running['id'])
    execute_acoustic_claim(jobs,claim,claim['worker_id'],threading.Event(),on_started=started)
    assert jobs.get(owner,claim['id'])['state']=='cancelled'
    assert handles
    for handle in handles:
        try:assert wait(handle,3000)==0
        finally:close(handle)
    checks['running_cancel_reaps_actual_child']=True
    # Lease expiry rejects the original publisher; explicit retry gets a new job.
    leased=track(batches.submit(owner,same.model_copy(update={'idempotency_key':uuid4().hex})))
    claim=jobs.claim('m01-'+suffix);identity=(claim['id'],claim['worker_id'],claim['generation'])
    with jobs.transaction() as tx:tx.execute('UPDATE {jobs} SET lease_until=? WHERE id=?',(tx.now()-1,claim['id']))
    from ptb_api.quota import StorageError
    try:files.output(identity,'late.wav','result',10)
    except StorageError as e:assert e.code=='stale_worker'
    else:raise AssertionError('Stale publisher accepted')
    files.fail(identity,'execution_failed');assert jobs.get(owner,claim['id'])['state']=='interrupted'
    replacement=jobs.retry(owner,claim['id'],uuid4().hex);assert replacement['retry_of']==claim['id']
    claim=jobs.claim('m01-'+suffix);execute_acoustic_claim(jobs,claim,claim['worker_id'],threading.Event())
    assert batches.get(owner,leased['id'])['summary']['complete']
    checks['expired_lease_stale_output_and_explicit_retry']=True
    # Fault injection at the second result reservation must not publish the first.
    partial=track(batches.submit(owner,same.model_copy(update={'idempotency_key':uuid4().hex})))
    claim=jobs.claim('m01-'+suffix);original=files.output;count=[0]
    def fail_second(identity,name,kind,expected):
        if kind=='result':
            count[0]+=1
            if count[0]==2:raise StorageError('quota_exceeded',413)
        return original(identity,name,kind,expected)
    files.output=fail_second
    try:execute_acoustic_claim(jobs,claim,claim['worker_id'],threading.Event())
    finally:files.output=original
    failed_job=jobs.get(owner,claim['id']);assert failed_job['state']=='failed' and failed_job['result_manifest'] is None
    with storage._locked() as conn:
        rows=FilePipeline._outputs(files,conn,(claim['id'],claim['worker_id'],claim['generation']))
        assert rows and all(a['state']=='deleted' and a['size_bytes']==0 and a['reserved_bytes']==0 for a in rows)
    checks['injected_second_reservation_failure_cleans_whole_output']=True
    # Deletion while an owned child exists invalidates the actual live input.
    deleted_ref=dict(audio=upload(wav,'delete-live.wav'),textgrid=tg)
    deleting=track(batches.submit(owner,BatchRequest(operation='textgrid_segment',project_id=project,idempotency_key=uuid4().hex,inputs=[deleted_ref],layer='syllable')))
    claim=jobs.claim('m01-'+suffix)
    execute_acoustic_claim(jobs,claim,claim['worker_id'],threading.Event(),on_started=lambda _:storage.delete(owner,deleted_ref['audio']['asset_id']))
    cancelled_job=jobs.get(owner,claim['id']);assert cancelled_job['state'] in ('cancelled','failed') and cancelled_job['result_manifest'] is None
    checks['actual_input_delete_during_child_blocks_publication']=True
    expired_ref=dict(audio=upload(wav,'expires-before-claim.wav'),textgrid=tg)
    expiring=track(batches.submit(owner,BatchRequest(operation='textgrid_segment',project_id=project,idempotency_key=uuid4().hex,inputs=[expired_ref],layer='syllable')))
    with storage._locked() as conn:
        conn.execute('UPDATE ptb_storage.assets SET expires_at=%s WHERE id=%s AND owner_id=%s',(storage._now(conn)-1,expired_ref['audio']['asset_id'],owner))
    assert jobs.claim('m01-'+suffix) is None
    assert batches.get(owner,expiring['id'])['summary']['counts']['failed']==1
    storage.delete(owner,expired_ref['audio']['asset_id'])
    checks['expired_input_before_claim_is_failed']=True
    with storage._locked() as conn:
        assert not conn.execute("SELECT 1 FROM ptb_storage.assets WHERE owner_id=%s AND state='uploading'",(owner,)).fetchone()
        after={t:[dict(r) for r in conn.execute('SELECT * FROM '+t)] for t in TABLES}
    for table,old in before.items():
        old_keys={json.dumps(r,sort_keys=True,default=str) for r in old}
        new_keys={json.dumps(r,sort_keys=True,default=str) for r in after[table]}
        assert old_keys<=new_keys,table
    assert storage.usage(owner)['reserved_bytes']==0
    checks['old_rows_preserved_and_no_reservation_leak']=True
    result=dict(checks=checks,batch_ids=[created['id'],cancelled['id'],failed['id']],test_people=people)
    (out/'pg-jobs.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'checks':checks}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--approved-m01-synthetic-tests',action='store_true');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.approved_m01_synthetic_tests:parser.error('Explicit synthetic test approval required')
    out=args.output.resolve()
    if not out.is_relative_to((ROOT/'output/validation/m01').resolve()) or not out.is_dir():parser.error('Unowned output')
    verify(json.loads(sys.stdin.readline()),out)
