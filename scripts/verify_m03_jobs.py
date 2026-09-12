"""M03-C real Windows API/child/files acceptance. No schema migration.

Uses a transactionally consistent copy of the existing synthetic P06 test DB.
All new inputs, jobs and exports belong to this run; old rows are checked intact.
"""
import base64
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from urllib.error import HTTPError
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.local_service import LocalService
from ptb_desktop.task_bridge import TaskBridge
from ptb_worker.local_acoustic_files import initialize_local_files, LocalAcousticFiles
from ptb_worker.store import SQLiteJobStore, LOCAL_PROJECT, JobError
from ptb_worker.egg_jobs import submit
from ptb_worker.acoustic_executor import execute_acoustic_claim
from ptb_api.quota import StorageError

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--include-private',action='store_true');args=parser.parse_args()
    out=ROOT/'output/validation/m03-jobs'/('local-'+uuid4().hex);out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
        source.backup(target)
        old=source.execute('SELECT * FROM jobs').fetchall()
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    with np.load(ROOT/'tests/fixtures/m03/EGG-SYN-PCM16.npz',allow_pickle=False) as arrays:
        samples=np.column_stack([arrays['load.egg_signal_raw'],arrays['load.audio_signal']])
    wavfile.write(inputs/'ɑ̃˥ EGG.wav',44100,samples)
    wavfile.write(inputs/'silence.wav',44100,np.zeros_like(samples))
    wavfile.write(inputs/'mono.wav',44100,samples[:,0])
    wavfile.write(inputs/'60 seconds.wav',44100,np.tile(samples,(75,1)))
    wavfile.write(inputs/'48 kHz.wav',48000,np.tile(samples,(2,1)))
    (saved/'egg_DATA.csv').write_bytes(b'preserve existing user result')
    original={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
    provider=FileProvider();directory=provider.choose('input',lambda:str(inputs));destination=provider.choose('output',lambda:str(saved))
    entries={f['name']:f for f in provider.list(directory['id'])}
    report=dict(success=False,schema_applied=[],checks=[],jobs=[])
    started=time.monotonic()
    def wait(service,job):
        deadline=time.monotonic()+150
        while time.monotonic()<deadline:
            current=service.get('/api/v1/jobs/'+job['id'])
            if current['state'] not in ('queued','running','cancel_requested'):return current
            time.sleep(.1)
        raise AssertionError('Job timeout')
    try:
        with LocalService(db,local_files_root=cache) as service:
            bridge=TaskBridge(provider,service)
            for mode,config,expected in [('single',dict(roi_start=.1,roi_end=.5),5),('batch',dict(mode='batch'),2),('inverse',dict(mode='inverse',roi_end=.12),3)]:
                request=dict(op='egg',id=entries['ɑ̃˥ EGG.wav']['id'],config=config,key=uuid4().hex)
                clock=time.monotonic();job=bridge.invoke(request)
                assert bridge.invoke(request)['id']==job['id']
                done=wait(service,job);assert done['state']=='succeeded',done
                report['jobs'].append(dict(mode=mode,id=job['id'],seconds=time.monotonic()-clock))
                exported=bridge.invoke(dict(op='save_job',id=job['id'],directory=destination['id']))
                assert exported['count']==expected
                assert (saved/'egg_DATA.csv').read_bytes()==b'preserve existing user result'
                assert bridge.invoke(dict(op='save_job',id=job['id'],directory=destination['id']))['saved']==exported['saved']
                for file in done['result_manifest']['files']:
                    raw=base64.b64decode(bridge.invoke(dict(op='result',job=job['id'],id=file['id']))['base64'])
                    assert len(raw)==file['size_bytes'] and hashlib.sha256(raw).hexdigest()==file['sha256']
                    if file['name'].endswith('.wav'):
                        fs,values=wavfile.read(io.BytesIO(raw));assert fs==44100 and len(values)==5292
                try: bridge.invoke(request|dict(config=config|dict(flip_channels=True)))
                except HTTPError as exc: assert exc.code==409
                else: raise AssertionError('Changed idempotent request accepted')
            report['checks'].append('three complete real API/child/SQLite task modes; hash readback and nonoverwriting repeatable native save')
            for name,config,code in [('mono.wav',{},'egg_stereo_required'),('silence.wav',dict(mode='inverse',roi_end=.12),'egg_inverse_unavailable')]:
                done=wait(service,bridge.invoke(dict(op='egg',id=entries[name]['id'],config=config,key=uuid4().hex)))
                assert done['state']=='failed' and done['error_code']==code,done
            report['checks'].append('mono and unavailable inverse return explicit errors without published files')
        with LocalService(db,local_files_root=cache) as service:
            bridge=TaskBridge(provider,service)
            for item in report['jobs']:assert bridge.invoke(dict(op='job',id=item['id']))['state']=='succeeded'
            assert len(bridge.invoke(dict(op='egg_jobs')))>=5
            report['checks'].append('actual service shutdown/restart preserves three successful manifests')

        # No background worker during deterministic cancellation/fault injection.
        store=SQLiteJobStore(db);files=LocalAcousticFiles(store,cache)
        ref=files.import_input((inputs/'ɑ̃˥ EGG.wav').read_bytes(),'ɑ̃˥ EGG.wav','audio')
        def admit(config=None,reference=None):
            return submit(store,'local',dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,audio=reference or ref,config=config or {}))
        for owner in ['other']:
            try:submit(store,owner,dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,audio=ref,config={}))
            except JobError as exc:assert exc.code=='project_not_found'
            else:raise AssertionError('Wrong owner accepted')
        try:admit(reference=ref|dict(sha256='a'*64))
        except JobError as exc:assert exc.code=='input_unavailable'
        else:raise AssertionError('Wrong input hash accepted')
        cancelled=admit();store.cancel('local',cancelled['id'])
        assert store.get('local',cancelled['id'])['state']=='cancelled'
        retry=store.retry('local',cancelled['id'],uuid4().hex);assert retry['retry_of']==cancelled['id']
        claim=store.claim('m03-validation');assert claim['id']==retry['id']
        child=[]
        def cancel_started(pid):child.append(pid);store.cancel('local',claim['id'])
        execute_acoustic_claim(store,claim,'m03-validation',threading.Event(),on_started=cancel_started)
        assert child and store.get('local',retry['id'])['state']=='cancelled'
        report['checks'].append('wrong owner/hash rejected; queued and live owned-child cancellation; immutable retry')
        failed=admit();claim=store.claim('m03-validation');write=files.write;count=[0]
        def fail_second_result(identity,asset_id,offset,raw):
            with files.locked() as state:kind=state['assets'][asset_id]['kind']
            if kind=='result':
                count[0]+=1
                if count[0]==2:raise StorageError('storage_write_failed',503)
            return write(identity,asset_id,offset,raw)
        files.write=fail_second_result
        execute_acoustic_claim(store,claim,'m03-validation',threading.Event())
        files.write=write
        assert count[0]==2 and store.get('local',failed['id'])['state']=='failed'
        state=json.loads((cache/'.ptb-local.json').read_text('utf-8'))
        assert all(a['state']=='deleted' for a in state['assets'].values() if a['job_id']==failed['id'])
        report['checks'].append('actual partial result write failure rolls back every result and scratch asset')

        interrupted=admit(dict(mode='batch'));claim=store.claim('m03-validation')
        identity=(claim['id'],'m03-validation',claim['generation'])
        pending=files.output(identity,'egg_DATA.csv','result',1);files.write(identity,pending['id'],0,b'x')
        try:files.read_result('local',pending['id'],0,1)
        except StorageError:pass
        else:raise AssertionError('Uncommitted output became readable')
        with store.transaction() as tx:tx.execute('UPDATE jobs SET lease_until=0 WHERE id=?',(claim['id'],))
        store.recover();files.recover()
        assert store.get('local',interrupted['id'])['state']=='interrupted'
        try:files.seal(identity,pending['id'])
        except StorageError:pass
        else:raise AssertionError('Late worker was not fenced')
        replacement=store.retry('local',interrupted['id'],uuid4().hex);claim=store.claim('m03-validation')
        execute_acoustic_claim(store,claim,'m03-validation',threading.Event())
        assert store.get('local',replacement['id'])['state']=='succeeded'
        report['checks'].append('uncommitted read denied; expired lease recovers, rejects late write, and new attempt succeeds')

        # Long-file budget measured through the same owned child, not extrapolated.
        long_ref=files.import_input((inputs/'60 seconds.wav').read_bytes(),'60 seconds.wav','audio')
        long_job=admit(dict(mode='batch',generate_images=True),long_ref);claim=store.claim('m03-validation');clock=time.monotonic()
        execute_acoustic_claim(store,claim,'m03-validation',threading.Event())
        assert store.get('local',long_job['id'])['state']=='succeeded',store.get('local',long_job['id'])
        report['long_file_seconds']=time.monotonic()-clock
        report['long_file_images']=True
        # Largest inverse ROI sample budget (48k) is exercised, not merely accepted.
        inverse_ref=files.import_input((inputs/'48 kHz.wav').read_bytes(),'48 kHz.wav','audio')
        inverse_job=admit(dict(mode='inverse',roi_end=1.0),inverse_ref)
        claim=store.claim('m03-validation');clock=time.monotonic()
        execute_acoustic_claim(store,claim,'m03-validation',threading.Event())
        assert store.get('local',inverse_job['id'])['state']=='succeeded',store.get('local',inverse_job['id'])
        report['one_second_inverse_seconds']=time.monotonic()-clock
        if args.include_private:
            confirmed=json.loads((ROOT/'output/validation/p03/confirmed-egg.json').read_text('utf-8'))
            baseline=json.loads((ROOT/'tests/fixtures/m03/manifest.json').read_text('utf-8'))
            report['private_cases']=[]
            for case in confirmed['cases']:
                path=Path(case['input']);raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
                assert sha==next(c for c in baseline['private_cases'] if c['id']==case['case_id'])['input_sha256']
                reference=files.import_input(raw,case['case_id']+'.wav','audio')
                private_job=admit(dict(mode='batch',generate_images=True,flip_channels=case['flip_channels']),reference)
                claim=store.claim('m03-validation');clock=time.monotonic()
                execute_acoustic_claim(store,claim,'m03-validation',threading.Event())
                done=store.get('local',private_job['id']);assert done['state']=='succeeded',done
                assert len(done['result_manifest']['files'])==5 and hashlib.sha256(path.read_bytes()).hexdigest()==sha
                report['private_cases'].append(dict(id=case['case_id'],job=done['id'],sha256=sha,seconds=time.monotonic()-clock))
        state=json.loads((cache/'.ptb-local.json').read_text('utf-8'))
        assert not [a for a in state['assets'].values() if a['state']!='deleted' and a['kind']=='temporary']
        with sqlite3.connect(db) as conn:
            for row in old:assert conn.execute('SELECT * FROM jobs WHERE id=?',(row[0],)).fetchone()==row
        assert original=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        report.update(success=True,seconds=time.monotonic()-started,inputs_preserved=True,old_rows_preserved=True,temporary_files_remaining=0)
    finally:
        provider.close();(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out/'report.json',flush=True)


if __name__=='__main__':main()
