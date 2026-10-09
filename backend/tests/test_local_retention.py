"""Expiry with real SQLite/asset locks in disposable owned fixtures."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4
import pytest
from ptb_worker.store import SQLiteJobStore, canonical, LOCAL_PROJECT, JobError
from ptb_worker.local_acoustic_files import LocalAcousticFiles, initialize_local_files
from ptb_worker.local_retention import LocalRetention, DAY
from ptb_api.quota import StorageError

ROOT = Path(__file__).resolve().parents[2]
NOW = int(time.time())

@pytest.fixture
def storage(tmp_path):
    db = tmp_path/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source, sqlite3.connect(db) as target:
        source.backup(target)
    store=SQLiteJobStore(db); root=tmp_path/'cache'; root.mkdir();initialize_local_files(root)
    files=LocalAcousticFiles(store,root)
    return store, files, LocalRetention(files)

def add(storage, *, age=31, operation='acoustic_analysis', count=1, legacy=False, inputs=(), parent=None, state='succeeded'):
    store,files,_=storage; job=str(uuid4()); ids=[str(uuid4()) for _ in range(count if state=='succeeded' else 0)]
    raw=b'owned-result'; sha=hashlib.sha256(raw).hexdigest()
    manifest=dict(files=[dict(id=key,name='result'+str(i)+'.wav',size_bytes=len(raw),sha256=sha) for i,key in enumerate(ids)])
    snapshot=dict(operation=operation, input_assets=[dict(id=key) for key in inputs])
    if parent:snapshot['analysis_job_id']=parent
    with files.batch_transaction() as tx:
        tx.execute("INSERT INTO {jobs}(id,owner_id,project_id,idempotency_key,request_hash,snapshot,state,deadline,created_at,updated_at,result_manifest) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   (job,'local',LOCAL_PROJECT,uuid4().hex,'0'*64,canonical(snapshot),state,NOW+1000,NOW-age*DAY,NOW-age*DAY,canonical(manifest) if state=='succeeded' else None))
        for i,key in enumerate(ids):
            a=dict(id=key,name='result'+str(i)+'.wav',role=None,kind='result',state='ready',sha256=sha,size_bytes=len(raw),reserved_bytes=0,expected_bytes=len(raw),job_id=job,generation=0)
            if not legacy:a['retention_created_at']=NOW-age*DAY
            files.state['assets'][key]=a;files._path(key).write_bytes(raw)
        files._save(files.state)
    return job,ids

def alive(storage,key):
    with storage[1].locked() as state:return state['assets'][key]['state']=='ready'

def test_old_unknown_assets_get_full_grace(storage):
    _,ids=add(storage,age=300,legacy=True)
    assert storage[2].sweep(now=NOW,force=True)['count']==0
    assert storage[2].sweep(now=NOW+30*DAY-1,force=True)['count']==0
    assert storage[2].sweep(now=NOW+30*DAY,force=True)['count']==1

def test_only_owned_expired_results_removed(storage,tmp_path):
    _,ids=add(storage)
    outside=tmp_path/'original.wav';outside.write_bytes(b'original-input')
    before=hashlib.sha256(outside.read_bytes()).hexdigest()
    result=storage[2].sweep(now=NOW,force=True)
    assert result['count']==1 and result['bytes']==len(b'owned-result')
    assert not alive(storage,ids[0]) and not storage[1]._path(ids[0]).exists()
    assert hashlib.sha256(outside.read_bytes()).hexdigest()==before
    with pytest.raises(StorageError,match='input_unavailable'):
        storage[1].read_result('local',ids[0],0,1)
    assert storage[2].sweep(now=NOW,force=True)['count']==0

def test_disable_and_interval(storage):
    _,ids=add(storage)
    storage[2].configure(enabled=False,days=30)
    assert storage[2].sweep(now=NOW,force=True)['skipped'] and alive(storage,ids[0])
    storage[2].configure(enabled=True,days=60)
    assert storage[2].sweep(now=NOW,force=True)['count']==0
    assert storage[2].sweep(now=NOW+1)['skipped']
    with pytest.raises(JobError):storage[2].configure(enabled=True,days=0)


def test_explicit_all_caches_ignores_age_but_retains_original_capture(storage,tmp_path):
    _,fresh=add(storage,age=0,count=2)
    _,captured=add(storage,age=0,operation='lip_analysis')
    original=tmp_path/'research.wav';original.write_bytes(b'original')
    storage[2].configure(enabled=False,days=3650)
    result=storage[2].sweep(now=NOW,all_cache=True)
    assert result['complete'] and result['count']==2
    assert all(not alive(storage,key) for key in fresh)
    assert alive(storage,captured[0]) and original.read_bytes()==b'original'


def test_all_caches_rejects_active_tasks_before_deletion(storage):
    _,fresh=add(storage,age=0)
    add(storage,state='running')
    with pytest.raises(JobError,match='local_cache_tasks_active'):storage[2].sweep(now=NOW,all_cache=True)
    assert alive(storage,fresh[0])

def test_recent_read_keeps_complete_bundle(storage):
    _,ids=add(storage,count=2)
    with storage[1].locked() as state:
        state['assets'][ids[0]]['retention_touched_at']=NOW-DAY;storage[1]._save(state)
    assert storage[2].sweep(now=NOW,force=True)['count']==0
    assert all(alive(storage,key) for key in ids)

def test_export_receipt_shortens_duplicate_cache_only(storage):
    _,ids=add(storage,age=10,count=2)
    storage[2].exported([ids[0]],now=NOW-8*DAY)
    assert storage[2].sweep(now=NOW,force=True)['count']==0
    storage[2].exported([ids[1]],now=NOW-8*DAY)
    assert storage[2].sweep(now=NOW,force=True)['count']==2

def test_active_consumer_and_parent_job_chain_are_protected(storage):
    parent,ids=add(storage)
    child,child_ids=add(storage,inputs=ids,age=31)
    add(storage,state='queued',parent=child)
    result=storage[2].sweep(now=NOW,force=True)
    assert result['count']==0 and result['protected_count']>=2
    assert all(alive(storage,key) for key in ids+child_ids)

def test_young_result_protects_analysis_job(storage):
    parent,ids=add(storage,operation='phonation_synthesis')
    add(storage,operation='phonation_synthesis',age=2,parent=parent)
    assert storage[2].sweep(now=NOW,force=True)['count']==0 and alive(storage,ids[0])

def test_recording_outputs_not_automatically_expired(storage):
    _,ids=add(storage,operation='lip_analysis',age=300)
    assert storage[2].sweep(now=NOW,force=True)['count']==0 and alive(storage,ids[0])

def test_hardlink_rejected_without_touching_original(storage,tmp_path):
    _,ids=add(storage)
    asset=storage[1]._path(ids[0]);outside=tmp_path/'protected.wav';os.link(asset,outside)
    with pytest.raises(StorageError,match='local_path_rejected'):
        storage[2].sweep(now=NOW,force=True)
    assert outside.read_bytes()==b'owned-result' and alive(storage,ids[0])

def test_other_instance_metadata_fails_closed(storage):
    _,ids=add(storage)
    storage[2].status(now=NOW)
    p=storage[1].root/'.ptb-retention.json';value=json.loads(p.read_text('utf8'))
    value['instance_id']=str(uuid4());p.write_text(json.dumps(value),'utf8')
    with pytest.raises(JobError,match='local_retention_metadata_rejected'):
        storage[2].sweep(now=NOW,force=True)
    assert alive(storage,ids[0])

def test_complete_candidate_set_is_checked_before_any_delete(storage):
    _,ids=add(storage,count=2)
    storage[1]._path(sorted(ids)[-1]).write_bytes(b'externally-changed')
    with pytest.raises(JobError,match='local_result_changed'):storage[2].sweep(now=NOW,force=True)
    assert all(alive(storage,key) and storage[1]._path(key).exists() for key in ids)

def test_access_denied_is_partial_and_keeps_retryable_file(storage,monkeypatch):
    _,ids=add(storage,count=2);blocked=ids[0];original=Path.unlink
    def denied(path,*args,**kwargs):
        if path.name==blocked+'.bin':raise PermissionError('QA access denial')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',denied)
    result=storage[2].sweep(now=NOW,force=True)
    assert result['count']==1 and result['failed_count']==1 and result['complete'] is False
    assert alive(storage,blocked) and storage[1]._path(blocked).exists()
    monkeypatch.setattr(Path,'unlink',original)
    assert storage[2].sweep(now=NOW,force=True)['count']==1

def test_local_api_requires_token_and_origin(storage):
    from fastapi.testclient import TestClient
    from ptb_api.main import create_app
    token='test-local-storage-token-1234567890';origin='http://127.0.0.1:5173'
    app=create_app('local',job_store=storage[0],local_token=token,local_origin=origin)
    headers={'Authorization':'Bearer '+token,'Origin':origin}
    with TestClient(app) as client:
        assert client.get('/api/v1/jobs/local-storage').status_code==403
        assert client.get('/api/v1/jobs/local-storage',headers=headers).status_code==200
        body={'enabled':True,'days':30,'exported_cache_days':7}
        assert client.post('/api/v1/jobs/local-storage/policy',json=body,headers={'Authorization':'Bearer '+token}).status_code==403
        assert client.post('/api/v1/jobs/local-storage/policy',json={**body,'path':'outside'},headers=headers).status_code==422
        assert client.post('/api/v1/jobs/local-storage/policy',json=body,headers=headers).status_code==200
