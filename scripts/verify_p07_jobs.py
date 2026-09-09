"""Real P07 joint acceptance in the approved database and marked generated-file root."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from unittest.mock import patch
from uuid import uuid4
import zipfile

import psycopg
from pydantic import ValidationError
from ptb_api.account_store import PostgresAccountStore
from ptb_api.storage import Storage
from ptb_api.storage_models import UploadInput
from ptb_api.job_models import FileJobInput
from ptb_api.quota import StorageError,CHUNK_BYTES
from ptb_worker.files import FilePipeline
from ptb_worker.store import PostgresJobStore
from ptb_worker.executor import execute_claim

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/validation/p07'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-p07-schema-and-test-files',action='store_true')
    if not parser.parse_args().approved_p07_schema_and_test_files: parser.error('Requires reviewed generated-file tests')
    config=json.loads(sys.stdin.readline())
    accounts=PostgresAccountStore(config['dsn'])
    jobs=PostgresJobStore(config['dsn'],lease_seconds=2)
    storage=Storage(config['dsn'],OUT/'storage',min_free_bytes=0)
    files=FilePipeline(jobs,storage)
    storage.recover()
    with storage._locked() as conn:
        assert conn.execute('SELECT current_database() AS n').fetchone()['n']=='ptb_p05_test_20260909'
        assert not conn.execute("SELECT 1 FROM ptb_jobs.jobs WHERE state IN ('queued','running','cancel_requested')").fetchone(), 'Existing active jobs must be left to their owner'
    people=[]; checks={}; owned_jobs=set(); worker=None
    suffix=uuid4().hex[:10]
    for index in range(2):
        user=accounts.create_user(f'p07j_{suffix}_{index}',secrets.token_urlsafe(32))
        project=accounts.create_project(str(user['id']),'P07 任务文件验证')
        people.append((str(user['id']),str(project['id'])))
    owner,project=people[0]
    def upload(data,name='part.bin',person=None):
        who,where=person or people[0]
        asset=storage.create(who,UploadInput(project_id=where,name=name,expected_bytes=len(data),idempotency_key=uuid4().hex))
        for offset in range(0,len(data),CHUNK_BYTES): storage.append(who,asset['id'],offset,data[offset:offset+CHUNK_BYTES])
        return storage.finalize(who,asset['id'])
    def submit(operation='storage_check',inputs=(),**options):
        body=FileJobInput(project_id=project,idempotency_key=uuid4().hex,operation=operation,config={'inputs':list(inputs),**options})
        job=jobs.submit(owner,body);owned_jobs.add(job['id']);return job
    def claim(job):
        value=jobs.claim('joint-'+suffix)
        assert value and value['id']==job['id'], 'Unexpected claimed job'
        return value,(value['id'],value['worker_id'],value['generation'])
    def run(job):
        value,_=claim(job)
        execute_claim(jobs,value,value['worker_id'],threading.Event())
        return jobs.get(owner,job['id'])
    def content(asset):
        result=bytearray()
        for offset in range(0,asset['size_bytes'],CHUNK_BYTES):result.extend(storage.read_block(owner,asset['id'],offset,min(CHUNK_BYTES,asset['size_bytes']-offset)))
        return bytes(result)
    def expect(code,action):
        try: action()
        except StorageError as error: assert error.code==code,(code,error.code)
        else: raise AssertionError('Expected '+code)
    def zip_bytes(name,data,compression=zipfile.ZIP_STORED):
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=compression) as archive:archive.writestr(name,data)
        return buffer.getvalue()
    try:
        source=upload(b'z'*100)
        with storage._locked() as conn:
            source_expiry=storage._now(conn)+3600
            conn.execute('UPDATE ptb_storage.assets SET expires_at=%s WHERE id=%s',(source_expiry,source['id']))
        first_job=submit(inputs=[source['id']],probe_bytes=777,probe_files=2)
        assert PostgresJobStore(config['dsn']).claim('metadata-only-worker') is None
        generated=run(first_job)
        assert generated['state']=='succeeded',generated
        outputs=generated['result_manifest']['files']
        assert len(outputs)==2
        for index,item in enumerate(outputs):
            expected=(bytes((n+index)%256 for n in range(256))*4)[:777]
            assert content(item)==expected and item['sha256']==hashlib.sha256(expected).hexdigest()
            assert item['expires_at']>source_expiry+6*86400
            expect('worker_owned_asset',lambda:storage.append(owner,item['id'],777,b'x'))
            expect('worker_owned_asset',lambda:storage.finalize(owner,item['id']))
        checks['Q04_Q20_two_results_exact_independent_ttl_and_public_write_rejection']=True

        archived=run(submit('archive_zip',[source['id']]))
        assert archived['state']=='succeeded',archived
        item=archived['result_manifest']['files'][0]
        assert item['expires_at']==source_expiry
        with zipfile.ZipFile(io.BytesIO(content(item))) as archive:
            assert archive.namelist()==['01-part.bin'] and archive.read('01-part.bin')==b'z'*100
        unpacked=run(submit('extract_zip',[item['id']]))
        assert unpacked['state']=='succeeded',unpacked
        extracted=unpacked['result_manifest']['files'][0]
        assert content(extracted)==b'z'*100 and extracted['expires_at']==source_expiry
        checks['Q14_Q20_zip_roundtrip_and_no_renewal']=True
        long_name='n'*176+'.bin'
        long_input=upload(b'long-name',long_name)
        long_archive=run(submit('archive_zip',[long_input['id']]))
        assert long_archive['state']=='succeeded'
        long_extract=run(submit('extract_zip',[long_archive['result_manifest']['files'][0]['id']]))
        assert long_extract['state']=='succeeded'
        long_file=long_extract['result_manifest']['files'][0]
        assert long_file['name']==long_name and content(long_file)==b'long-name'
        checks['maximum_filename_zip_roundtrip']=True

        for name,data in [('../escape',b'x'),('bomb',b'0'*300000)]:
            bad=upload(zip_bytes(name,data,zipfile.ZIP_DEFLATED),'bad.zip')
            before=storage.usage(owner)
            result=run(submit('extract_zip',[bad['id']]))
            assert result['state']=='failed' and result['error_code']=='archive_rejected',result
            assert result['result_manifest'] is None
            assert storage.usage(owner)['used_bytes']==before['used_bytes']
        # Falsified directory size is rejected before allocating that directory.
        import struct
        bad_data=bytearray(zip_bytes('file',b'content'))
        struct.pack_into('<I',bad_data,bad_data.rfind(b'PK\x05\x06')+12,100_000_000)
        bad=upload(bytes(bad_data),'directory-bomb.zip')
        assert run(submit('extract_zip',[bad['id']]))['error_code']=='archive_rejected'
        checks['Q03_zip_path_ratio_and_directory_bombs']=True

        for operation,inputs,options in [('storage_check',[],{'probe_bytes':16,'probe_files':2,'max_output_bytes':16}),
                                          ('archive_zip',[source['id']],{'max_output_bytes':160})]:
            before=storage.usage(owner)
            result=run(submit(operation,inputs,**options))
            assert result['state']=='failed' and result['error_code']=='output_budget_exceeded',result
            assert result['result_manifest'] is None
            after=storage.usage(owner)
            assert (before['used_bytes'],before['reserved_bytes'])==(after['used_bytes'],after['reserved_bytes'])
        before=storage.usage(owner)
        fill=storage.create(owner,UploadInput(project_id=project,name='reserved.bin',expected_bytes=before['available_bytes']-20,idempotency_key=uuid4().hex))
        result=run(submit(probe_bytes=16,probe_files=2))
        assert result['state']=='failed' and result['error_code']=='quota_exceeded',result
        last=storage.create(owner,UploadInput(project_id=project,name='last-space.bin',expected_bytes=20,idempotency_key=uuid4().hex))
        expect('quota_exceeded',lambda:submit())
        storage.delete(owner,last['id'])
        storage.delete(owner,fill['id'])
        assert storage.usage(owner)['used_bytes']==before['used_bytes']
        checks['Q04_Q14_multi_output_and_zip_tail_budget_atomic_failure']=True

        other=upload(b'private',person=people[1])
        expect('asset_not_found',lambda:submit('archive_zip',[other['id']]))
        # Relational enforcement independently rejects even a direct bad insert.
        with storage._locked() as conn:
            try:
                with conn.transaction():
                    conn.execute("INSERT INTO ptb_storage.job_assets(job_id,asset_id,owner_id,project_id,role,generation,input_sha256,input_expires_at,created_at) VALUES(%s,%s,%s,%s,'input',0,%s,%s,%s)",
                        (generated['id'],other['id'],owner,project,other['sha256'],other['expires_at'],time.time()))
            except psycopg.errors.ForeignKeyViolation: pass
            else: raise AssertionError('Cross-owner FK accepted')
        try: FileJobInput(project_id=project,idempotency_key=uuid4().hex,operation='native_tool',config={'output_path':'arbitrary'})
        except ValidationError: pass
        else: raise AssertionError('Uncontrolled native path accepted')
        checks['Q12_database_owner_fk_Q04_native_path_rejected']=True

        active_source=upload(b'active')
        job=submit(inputs=[active_source['id']]);value,identity=claim(job)
        staged=files.output(identity,'staged.bin','result',3)
        files.write(identity,staged['id'],0,b'abc');files.seal(identity,staged['id'])
        expect('asset_unavailable',lambda:storage.metadata(owner,staged['id']))
        expect('stale_worker',lambda:files.complete((identity[0],identity[1],identity[2]-1)))
        with storage._locked() as conn:
            conn.execute('UPDATE ptb_storage.assets SET expires_at=%s WHERE id=%s',(storage._now(conn)-1,active_source['id']))
        expect('input_unavailable',lambda:files.complete(identity))
        expect('input_unavailable',lambda:files.read_input(identity,active_source['id'],0,1))
        storage.cleanup(); files.fail(identity,'input_unavailable')
        assert jobs.get(owner,job['id'])['state']=='cancelled'
        assert not storage._path(staged['id']).exists()
        checks['Q07_active_input_expiry_Q17_fenced_manifest_and_hidden_staging']=True

        cancelled=submit();_,cancel_identity=claim(cancelled)
        cancelled_file=files.output(cancel_identity,'cancelled.bin','result',1)
        files.write(cancel_identity,cancelled_file['id'],0,b'c');files.seal(cancel_identity,cancelled_file['id'])
        jobs.cancel(owner,cancelled['id'])
        expect('cancelled',lambda:files.complete(cancel_identity))
        files.fail(cancel_identity,'cancelled')
        assert jobs.get(owner,cancelled['id'])['state']=='cancelled'
        assert not storage._path(cancelled_file['id']).exists()
        checks['explicit_cancel_before_manifest_cannot_succeed']=True

        boundary_input=upload(b'expire-during-commit')
        boundary_job=submit(inputs=[boundary_input['id']]);_,boundary_identity=claim(boundary_job)
        boundary_output=files.output(boundary_identity,'commit-boundary.bin','result',1)
        files.write(boundary_identity,boundary_output['id'],0,b'b');files.seal(boundary_identity,boundary_output['id'])
        with storage._locked() as conn:
            conn.execute('UPDATE ptb_storage.assets SET expires_at=%s WHERE id=%s',(storage._now(conn)+.15,boundary_input['id']))
        real_path=storage._path
        class DelayedStat:
            def stat(self):
                time.sleep(.2)
                return real_path(boundary_output['id']).stat()
        def delay_stat(asset_id):
            return DelayedStat() if str(asset_id)==boundary_output['id'] else real_path(asset_id)
        with patch.object(storage,'_path',side_effect=delay_stat):
            expect('input_unavailable',lambda:files.complete(boundary_identity))
        assert jobs.get(owner,boundary_job['id'])['result_manifest'] is None
        assert next(a for a in storage.list(owner,project) if a['id']==boundary_output['id'])['state']=='uploading'
        files.fail(boundary_identity,'input_unavailable')
        checks['expiry_during_final_commit_rolls_back_all_visibility']=True

        # Completion vs deletion: only the serialized winner can publish.
        racing_source=upload(b'race')
        job=submit(inputs=[racing_source['id']]);value,identity=claim(job)
        staged=files.output(identity,'race-result.bin','result',4)
        files.write(identity,staged['id'],0,b'done');files.seal(identity,staged['id'])
        assert files.impact(owner,racing_source['id'])['active_jobs']==[job['id']]
        def finish_race():
            try:files.complete(identity)
            except StorageError as error:files.fail(identity,error.code)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(finish_race);second=pool.submit(storage.delete,owner,racing_source['id'])
            first.result();second.result()
        state=jobs.get(owner,job['id'])
        assert state['state'] in ('succeeded','cancelled')
        if state['state']=='succeeded': assert content(state['result_manifest']['files'][0])==b'done'
        else: assert state['result_manifest'] is None and not storage._path(staged['id']).exists()
        checks['Q18_generation_delete_race_atomic_visibility']=state['state']

        job=submit(probe_bytes=4*CHUNK_BYTES,probe_files=4)
        worker=subprocess.Popen([sys.executable,'-X','utf8','-m','ptb_worker.cli','--probe-step-delay','0.2'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        worker.stdin.write(json.dumps({'kind':'postgres','dsn':config['dsn'],'storage_root':str(storage.root),'lease_seconds':2})+'\n');worker.stdin.flush()
        assert json.loads(worker.stdout.readline())['ready']
        deadline=time.monotonic()+15;old=None
        while time.monotonic()<deadline:
            with storage._locked() as conn:
                row=conn.execute('SELECT * FROM ptb_jobs.jobs WHERE id=%s',(job['id'],)).fetchone()
                writes=conn.execute("SELECT a.* FROM ptb_storage.assets a JOIN ptb_storage.job_assets l ON l.asset_id=a.id WHERE l.job_id=%s AND l.role='output' AND a.size_bytes>0",(job['id'],)).fetchall()
            if writes and row['state']=='running':old=dict(row);break
            time.sleep(.05)
        assert old,'Worker produced no staged block before timeout'
        worker.terminate();worker.wait(5)
        worker.stdin.close();worker.stdout.close();worker.stderr.close();worker=None
        deadline=time.monotonic()+6
        while time.monotonic()<deadline:
            # Wait on database time, before invoking recovery. Calling get()
            # here would itself transition an expired job after file recovery.
            with storage._locked() as conn:
                lease=conn.execute('SELECT lease_until FROM ptb_jobs.jobs WHERE id=%s',(job['id'],)).fetchone()['lease_until']
                expired=storage._now(conn)>=lease
            if expired:break
            time.sleep(.1)
        assert expired
        storage.recover()
        assert jobs.get(owner,job['id'])['state']=='interrupted'
        old_identity=(old['id'],old['worker_id'],old['generation'])
        expect('stale_worker',lambda:files.complete(old_identity))
        expect('stale_worker',lambda:files.write(old_identity,str(writes[0]['id']),writes[0]['size_bytes'],b'x'))
        assert all(not storage._path(row['id']).exists() for row in writes)
        retry=jobs.retry(owner,job['id'],uuid4().hex);owned_jobs.add(retry['id'])
        result=run(retry)
        assert result['state']=='succeeded' and len(result['result_manifest']['files'])==4,result
        checks['Q08_Q10_owned_worker_terminated_recovered_Q17_old_worker_retry']=True
    finally:
        if worker is not None:
            worker.terminate();worker.wait(5)
            worker.stdin.close();worker.stdout.close();worker.stderr.close()
        for job_id in owned_jobs:
            jobs.cancel(owner,job_id)
        for who,where in people:
            for asset in storage.list(who,where):
                assert storage._path(asset['id']).resolve().parent==storage.root.resolve()
                storage.delete(who,asset['id'])
            assert storage.usage(who)['used_bytes']==storage.usage(who)['reserved_bytes']==0
    report={'scope':'Windows real PG + marked disk + owned worker joint file acceptance','checks':checks,
            'generated_files_removed':True,'hard_power_loss_tested':False}
    (OUT/'jobs-validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
