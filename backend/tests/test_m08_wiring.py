"""M08 formal adapters on new synthetic assets and a copied existing schema."""
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from uuid import uuid4
import pytest
from ptb_worker.store import SQLiteJobStore,LOCAL_PROJECT,JobError
from ptb_worker.local_acoustic_files import LocalAcousticFiles,initialize_local_files
from ptb_worker.acoustic_executor import execute_acoustic_claim
from ptb_worker.m08_task import submit
from ptb_worker.m08_results import listing,manage,save_copy
from ptb_api.m08_models import M08Manage
from ptb_api.quota import StorageError

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime(tmp_path):
    db=tmp_path/'tasks.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as destination:
        source.backup(destination)
    store=SQLiteJobStore(db);cache=tmp_path/'cache';cache.mkdir();initialize_local_files(cache)
    files=LocalAcousticFiles(store,cache)
    raw=(ROOT/'output/validation/m08-wiring/input.wav').read_bytes()
    ref=files.import_input(raw,'synthetic.wav','audio')
    return store,files,ref


def task(runtime,config=None):
    store,files,ref=runtime
    return submit(store,'local',dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,audio=ref,config=config or {'action':'transform'}))


def run(runtime,job,on_started=None,stop=None):
    store,files,_=runtime;claim=store.claim('m08-tests');assert claim['id']==job['id']
    evidence={};execute_acoustic_claim(store,claim,'m08-tests',stop or threading.Event(),on_started=on_started,process_evidence=evidence)
    return store.get('local',job['id']),evidence,claim


def clean(files):
    with files.locked() as state:
        assert not [a for a in state['assets'].values() if a['reserved_bytes'] or a['kind']=='temporary' and a['state']!='deleted']


def test_three_format_scientific_decode(runtime,record_property):
    store,files,_=runtime
    for ext in ('wav','mp3','flac'):
        ref=files.import_input((ROOT/f'output/validation/m08-wiring/input.{ext}').read_bytes(),'synthetic.'+ext,'audio')
        rt=(store,files,ref);job=task(rt,{'action':'preview'});result,evidence,_=run(rt,job)
        assert result['state']=='succeeded',result
        assert evidence['configured_memory_bytes']==1_000_000_000 and evidence['memory_peak_bytes']>0
        assert evidence['cleaned'];clean(files)
        record_property('m08_'+ext,json.dumps(evidence))


@pytest.mark.parametrize('owner,project,sha',[('other',LOCAL_PROJECT,None),('local',str(uuid4()),None),('local',LOCAL_PROJECT,'0'*64)])
def test_scope_rejection(runtime,owner,project,sha):
    store,_,ref=runtime
    with pytest.raises(JobError):
        submit(store,owner,dict(project_id=project,idempotency_key=uuid4().hex,audio=dict(ref,sha256=sha or ref['sha256']),config={'action':'transform'}))


def test_direct_linux_submission_is_closed(runtime,monkeypatch):
    import sys
    with monkeypatch.context() as context:
        context.setattr(sys,'platform','linux')
        with pytest.raises(JobError,match='m08_platform_unverified'):
            task(runtime)


def test_cancel_during_native_and_recovery(runtime):
    store,files,_=runtime;job=task(runtime);stop=threading.Event()
    result,evidence,_=run(runtime,job,on_started=lambda pid:stop.set(),stop=stop)
    assert result['state']=='cancelled';assert evidence['cleaned'];clean(files)
    result,_,_=run(runtime,task(runtime));assert result['state']=='succeeded';clean(files)


def test_partial_stream_failure_never_publishes(runtime,monkeypatch):
    store,files,_=runtime;original=files.write
    def fail(identity,asset_id,offset,raw):
        if files.state['assets'][asset_id]['kind']=='result':raise StorageError('storage_write_failed',507)
        return original(identity,asset_id,offset,raw)
    monkeypatch.setattr(files,'write',fail)
    result,evidence,_=run(runtime,task(runtime))
    assert result['state']=='failed' and result['result_manifest'] is None
    assert evidence['cleaned'];clean(files)


def test_timeout_cleanup(runtime,monkeypatch):
    from dataclasses import replace
    import ptb_worker.acoustic_executor as executor
    original=executor.collect_scientific
    def short(entry,request,scratch,limits,*args,**kwargs):
        return original(entry,request,scratch,replace(limits,timeout_seconds=.01),*args,**kwargs)
    monkeypatch.setattr(executor,'collect_scientific',short)
    result,evidence,_=run(runtime,task(runtime))
    assert result['state']=='failed' and result['error_code']=='deadline_exceeded'
    assert evidence['cleaned'];clean(runtime[1])


def test_late_worker_rejected(runtime):
    store,files,_=runtime;job=task(runtime);claim=store.claim('old-worker')
    with store.transaction() as tx:
        tx.execute("UPDATE {jobs} SET generation=generation+1,worker_id='new-worker' WHERE id=?",(job['id'],))
    identity=(job['id'],'old-worker',claim['generation'])
    with pytest.raises(StorageError,match='stale_worker'):files.output(identity,'late.wav','result',100)
    with pytest.raises(StorageError,match='stale_worker'):files.complete(identity)
    clean(files)


def test_native_crash_cleanup(runtime):
    import os,signal
    result,evidence,_=run(runtime,task(runtime),on_started=lambda pid:os.kill(pid,signal.SIGTERM))
    assert result['state']=='failed' and result['result_manifest'] is None
    assert evidence['cleaned'];clean(runtime[1])


def test_serial_linear_outputs_are_actual_pcm(runtime):
    import io
    from scipy.io import wavfile
    store,files,_=runtime
    config=dict(action='linear',start=0,end=1,points=[dict(time=.1,freqs=[120,180],mode='full'),dict(time=.8,freqs=[140,200],mode='full')])
    done,_,_=run(runtime,task(runtime,config))
    assert done['state']=='succeeded',done
    outputs=[f for f in done['result_manifest']['files'] if f['name'].endswith('.wav')]
    assert len(outputs)==4
    for file in outputs:
        raw=files.read_result('local',file['id'],0,file['size_bytes'])
        rate,data=wavfile.read(io.BytesIO(raw))
        assert rate==16000 and len(data)==16000 and data.dtype.name=='int16'
    clean(files)


def test_explicit_ids_name_preflight_and_numbering(runtime):
    store,files,ref=runtime;job=task(runtime);result,_,_=run(runtime,job)
    r=listing(store,'local',LOCAL_PROJECT)[0]['results'][0]
    scope=M08Manage(project_id=LOCAL_PROJECT,source=ref,ids=[r['id']])
    # Reserve two names before either copy runs; queued snapshots prevent races.
    a=save_copy(store,'local',scope);b=save_copy(store,'local',scope)
    run(runtime,a);run(runtime,b)
    values=[r for j in listing(store,'local',LOCAL_PROJECT) for r in j['results']]
    assert {v['name'] for v in values}=={'synthetic.wav','synthetic_0.00_1.00_modified_1.wav','synthetic_0.00_1.00_modified_2.wav'}
    with pytest.raises(JobError):manage(store,'local',scope.model_copy(update={'names':['../escape.wav']}),'rename')
    with pytest.raises(JobError):manage(store,'local',scope.model_copy(update={'names':['synthetic_0.00_1.00_modified_1.wav']}),'rename')
    with pytest.raises(JobError):manage(store,'local',scope.model_copy(update={'ids':[ref['asset_id']]}),'remove')
    assert manage(store,'local',scope,'remove')['removed']==[r['id']]
    clean(files)
