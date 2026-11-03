"""Review regressions use new SQLite fixtures, never the user's task database."""
import threading
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from ptb_api.job_models import JobInput
from ptb_worker.local_workspace import prepare_workspace
from ptb_worker.local_acoustic_files import LocalAcousticFiles
from ptb_worker.store import SQLiteJobStore, LOCAL_PROJECT, QUEUE_WAIT_SECONDS
from ptb_worker import executor


@pytest.fixture
def store(tmp_path):
    db, _ = prepare_workspace(tmp_path/'workspace', Path(__file__).parents[1]/'migrations')
    return SQLiteJobStore(db)


def submit(store):
    return store.submit('local', JobInput(project_id=LOCAL_PROJECT, idempotency_key=uuid4().hex))


def test_finished_history_does_not_consume_capacity(store):
    with store.transaction() as tx:
        for _ in range(1001):
            key=uuid4().hex
            tx.execute("INSERT INTO {jobs}(id,owner_id,project_id,idempotency_key,request_hash,snapshot,state,deadline,created_at,updated_at) VALUES(?,?,?,?,?,?,'failed',1,1,1)",
                       (key,'local',LOCAL_PROJECT,key,'0'*64,'{}'))
    assert submit(store)['state']=='queued'
    with store.transaction() as tx:assert store.capacity_used(tx,'local')==1


def test_queue_wait_does_not_consume_execution_budget(store):
    job=submit(store)
    with store.transaction() as tx:
        tx.execute('UPDATE {jobs} SET created_at=created_at-600,deadline=deadline-600 WHERE id=?',(job['id'],))
    assert store.get('local',job['id'])['state']=='queued'
    claim=store.claim('worker')
    with store.transaction(write=False) as tx:now=tx.now()
    assert 299 < claim['deadline']-now <= 300
    assert store.heartbeat(job['id'],'worker',claim['generation'],.1)=='running'


def test_queue_has_separate_bounded_wait(store):
    job=submit(store)
    with store.transaction() as tx:
        tx.execute('UPDATE {jobs} SET created_at=created_at-? WHERE id=?',(QUEUE_WAIT_SECONDS+1,job['id']))
    assert store.get('local',job['id'])['error_code']=='deadline_exceeded'


def test_worker_survives_poison_task_and_claims_next(store,monkeypatch):
    first=submit(store);second=submit(store);stop=threading.Event();seen=[]
    def execute(store,claim,worker,stop,**kwargs):
        seen.append(claim['id'])
        if len(seen)==1:raise KeyError('private snapshot field')
        stop.set()
    monkeypatch.setattr(executor,'execute_claim',execute)
    executor.run_worker(store,stop,'worker')
    assert set(seen)=={first['id'],second['id']}
    assert store.get('local',seen[0])['error_code']=='execution_failed'


@pytest.mark.parametrize('fs',[8000,11025,16000])
def test_spectral_band_outside_nyquist_is_missing(fs):
    from phonetic_core.acoustic.spectral_batch import compute_spectral_features_batch
    y=np.sin(2*np.pi*200*np.arange(fs//5)/fs)
    track=np.full(20,200.)
    values=compute_spectral_features_batch(y,fs,10,track,track*3,track*7,track*12,3)
    # Frame zero has no complete centred N-period window by the existing contract.
    assert np.isnan(values['H1'][0])
    assert np.isfinite(values['H1'][1:]).all()
    assert np.isnan(values['H5K']).all() if fs==8000 else np.isfinite(values['H5K'][1:]).all()


@pytest.mark.parametrize('error',[MemoryError, __import__('phonetic_core.ports.errors',fromlist=['BackendAborted']).BackendAborted])
def test_irapt_resource_abort_does_not_fall_back(error):
    from phonetic_core.acoustic.jitter_shimmer import _extract_f0_for_wm
    def abort(*args,**kwargs):raise error()
    with pytest.raises(error):_extract_f0_for_wm(np.zeros(800),8000,75,500,f0_provider=abort)


def test_deleted_asset_metadata_is_compacted(tmp_path):
    db,root=prepare_workspace(tmp_path/'workspace',Path(__file__).parents[1]/'migrations')
    files=LocalAcousticFiles(SQLiteJobStore(db),root)
    with files.locked() as state:
        state['assets']={str(uuid4()):dict(state='deleted',reserved_bytes=0) for _ in range(10000)}
        files._budget(1)
        files._save(state)
    with files.locked() as state:assert state['assets']=={}


def test_stream_chunks_recover_size_without_metadata_rewrites(tmp_path,monkeypatch):
    import hashlib
    db,root=prepare_workspace(tmp_path/'workspace',Path(__file__).parents[1]/'migrations')
    files=LocalAcousticFiles(SQLiteJobStore(db),root)
    key=files.begin_stream_input('table.ptb.sqlite','parameter_bundle',6)
    before=(root/'.ptb-local.json').read_bytes()
    files.append_stream_input(key,b'abc')
    reopened=LocalAcousticFiles(SQLiteJobStore(db),root)
    reopened.append_stream_input(key,b'def')
    assert (root/'.ptb-local.json').read_bytes()==before
    assert reopened.finish_stream_input(key)['sha256']==hashlib.sha256(b'abcdef').hexdigest()
    with reopened.locked() as state:
        assert state['assets'][key]['size_bytes']==6
        assert state['assets'][key]['reserved_bytes']==0
        assert state['assets'][key]['state']=='ready'


def test_pipe_drains_bytes_written_between_empty_read_and_exit(monkeypatch,tmp_path):
    from ptb_worker.native import windows
    from ptb_worker.native.reaper import collect_pipe
    from ptb_worker.io.limits import Limits
    class Process:
        pid=1
        def __init__(self,*args):pass
        def poll(self):return 0
        def close(self):pass
    class Pipe:
        def __init__(self):self.chunks=iter([b'',b'last bytes',b''])
        def read(self,*args):return next(self.chunks),False
        def close(self):pass
    monkeypatch.setattr(windows,'OwnedProcess',Process)
    assert collect_pipe([],Pipe(),tmp_path,Limits())[0]==b'last bytes'


def test_oversize_deferred_snapshot_becomes_failed_child_and_queue_continues(tmp_path,monkeypatch):
    import json
    from ptb_worker.acoustic_batches import AcousticBatches
    db,root=prepare_workspace(tmp_path/'workspace',Path(__file__).parents[1]/'migrations')
    jobs=SQLiteJobStore(db);files=LocalAcousticFiles(jobs,root);batches=AcousticBatches(jobs,files)
    ref=files.import_input(b'a,b\n1,2','fixture.csv','table')
    monkeypatch.setattr(files,'validate_batch_input',lambda *args:[dict(id=ref['asset_id'],role='audio',name='fixture.wav',extra='x'*20000)])
    result=batches.submit('local',dict(operation='acoustic_analysis',project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,inputs=[{'audio':ref}],config={}))
    child=result['summary']['items'][0]['job_id']
    assert jobs.get('local',child)['error_code']=='snapshot_too_large'
    next_job=submit(jobs)
    assert jobs.claim('next')['id']==next_job['id']
