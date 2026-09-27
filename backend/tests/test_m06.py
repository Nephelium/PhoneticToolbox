"""Real persistent tasks on an isolated copy of the existing SQLite schema."""
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import threading
from uuid import uuid4
import numpy as np
import pytest
from scipy.io import wavfile
from ptb_worker.store import SQLiteJobStore,LOCAL_PROJECT,JobError
from ptb_worker.local_acoustic_files import LocalAcousticFiles,initialize_local_files
from ptb_worker.m06_task import submit as submit_ref
from ptb_worker.m06_executor import execute_claim,unpack
from phonetic_core.synthesis.klatt.api import defaults,import_parameters,export_parameters

ROOT=Path(__file__).resolve().parents[2]


def submit(store,owner,request):
    request=dict(request)
    if 'config' in request:
        request['parameters']=store.files.import_input(export_parameters(request.pop('config')).encode(),'m06-parameters.csv','table')
    return submit_ref(store,owner,request)


@pytest.fixture
def runtime(tmp_path):
    db=tmp_path/'tasks.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as dest:source.backup(dest)
    store=SQLiteJobStore(db);cache=tmp_path/'cache';cache.mkdir();initialize_local_files(cache)
    files=LocalAcousticFiles(store,cache)
    return store,files


def body(action='synthesize',duration=.2):
    c=defaults();c['duration']=duration
    for curve in c['curves'].values():curve['points'][-1][0]=duration
    return dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,action=action,config=c)


def run(runtime,request,stop=None,on_started=None):
    store,files=runtime;job=submit(store,'local',request);claim=store.claim('m06-tests');assert claim['id']==job['id']
    evidence={};execute_claim(store,claim,'m06-tests',stop or threading.Event(),on_started=on_started,evidence=evidence)
    result=store.get('local',job['id'])
    with files.locked() as state:assert not [a for a in state['assets'].values() if a['reserved_bytes'] or a['kind']=='temporary' and a['state']!='deleted']
    return result,evidence


@pytest.mark.parametrize('action',['generate','synthesize','extract'])
def test_real_task_wav_snapshot(runtime,action):
    store,files=runtime;request=body(action);request['config']['sequence']='a i'
    if action=='extract':request['audio']=files.import_input((ROOT/'tests/fixtures/m06/source.wav').read_bytes(),'source.wav','audio')
    result,evidence=run(runtime,request)
    assert result['state']=='succeeded',result
    assert evidence['memory_peak_bytes']>0 and evidence['cleaned']
    outputs={f['name']:files.read_result('local',f['id'],0,f['size_bytes']) for f in result['result_manifest']['files']}
    for f in result['result_manifest']['files']:assert hashlib.sha256(outputs[f['name']]).hexdigest()==f['sha256']
    meta=json.loads(outputs['m06.ptb.json']);assert import_parameters(outputs['parameters.csv'].decode())==meta['config']
    if action=='synthesize':
        rate,samples=wavfile.read(io.BytesIO(outputs['synthesis.wav']))
        assert rate==16000 and samples.shape==(3200,) and samples.dtype==np.int16
        assert np.max(np.abs(samples))>0


def test_idempotency_owner_project(runtime):
    store,_=runtime;r=body();job=submit(store,'local',r);assert submit(store,'local',r)['id']==job['id']
    with pytest.raises(JobError):submit(store,'other',r)
    r['config']['fade_in']=1
    with pytest.raises(JobError,match='idempotency_conflict'):submit(store,'local',r)
    r=body();r['project_id']=str(uuid4())
    with pytest.raises(JobError):submit(store,'local',r)


def test_failure_and_running_cancel(runtime):
    request=body('generate');request['config']['sequence']='?'
    failed,_=run(runtime,request);assert failed['state']=='failed' and failed['error_code']=='m06_invalid_ipa'
    stop=threading.Event()
    cancelled,evidence=run(runtime,body(duration=10),stop,on_started=lambda pid:stop.set())
    assert cancelled['state']=='cancelled' and cancelled['result_manifest'] is None


def test_stale_generation_cannot_publish(runtime):
    store,files=runtime;job=submit(store,'local',body());claim=store.claim('stale')
    store.cancel('local',job['id']);execute_claim(store,claim,'stale',threading.Event())
    assert store.get('local',job['id'])['state']=='cancelled'
    with pytest.raises(Exception):files.output((job['id'],'stale',claim['generation']),'late.wav','result',8)


def test_bad_source_and_bound(runtime):
    store,files=runtime;r=body('extract');r['audio']=files.import_input(b'broken','broken.wav','audio')
    failed,_=run(runtime,r);assert failed['state']=='failed' and failed['error_code']=='m06_audio_decode_failed'
    failed,_=run(runtime,body(duration=10.01));assert failed['state']=='failed' and failed['error_code']=='m06_admission_budget'


def test_bundle_rejects_corruption():
    with pytest.raises(Exception):unpack(b'bad','0'*64,'synthesize')


def test_linux_gate_closed(runtime,monkeypatch):
    import ptb_worker.m06_task as module
    monkeypatch.setattr(module.sys,'platform','linux')
    with pytest.raises(JobError,match='m06_platform_unverified'):submit(runtime[0],'local',body())
