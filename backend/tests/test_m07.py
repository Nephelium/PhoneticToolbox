"""M07 real owned-process and shared persistent task checks."""
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
from ptb_worker.m07_task import submit
from ptb_worker.m07_executor import execute_claim
from ptb_api.m07_models import M07Request

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture
def runtime(tmp_path):
    db=tmp_path/'tasks.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as dest:source.backup(dest)
    store=SQLiteJobStore(db);cache=tmp_path/'cache';cache.mkdir();initialize_local_files(cache);files=LocalAcousticFiles(store,cache)
    refs=[]
    for i in range(2):
        raw=(ROOT/f'output/validation/m07/baseline/round1/input{i}.wav').read_bytes();refs.append(files.import_input(raw,f'input{i}.wav','audio'))
    return store,files,refs

def body(runtime,action='analyze',**kwargs):
    return dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,action=action,source=runtime[2][0],target=runtime[2][1],**kwargs)

def run(runtime,request,stop=None,on_started=None):
    store,files,_=runtime;job=submit(store,'local',request);claim=store.claim('m07-test');assert claim['id']==job['id']
    evidence={};execute_claim(store,claim,'m07-test',stop or threading.Event(),on_started=on_started,evidence=evidence)
    result=store.get('local',job['id'])
    with files.locked() as state:assert not [a for a in state['assets'].values() if a['reserved_bytes'] or a['kind']=='temporary' and a['state']!='deleted']
    return result,evidence

def outputs(runtime,result):
    return {f['name']:b''.join(runtime[1].read_result('local',f['id'],i,min(1048576,f['size_bytes']-i)) for i in range(0,f['size_bytes'],1048576)) for f in result['result_manifest']['files']}

def test_analysis_apply_six_groups(runtime):
    result,ev=run(runtime,body(runtime));assert result['state']=='succeeded',result
    assert ev['memory_peak_bytes']>0 and ev['cleaned']
    data=outputs(runtime,result);meta=json.loads(data['m07.ptb.json']);controls=meta['controls'];controls['source'][8]+=10
    applied,_=run(runtime,body(runtime,'apply',analysis_job_id=result['id'],controls=controls));assert applied['state']=='succeeded',applied
    before=json.loads(data['analysis.m07.json']);after=json.loads(outputs(runtime,applied)['analysis.m07.json'])
    assert before['source']['f0_hz']!=after['source']['f0_hz']
    for key in ('residual','pulses','lpc_coefficients'):assert before['source'][key]==after['source'][key]
    all_ids=[]
    for reverse in (False,True):
        for kind in (2,1,3):
            generated,evidence=run(runtime,body(runtime,'generate',analysis_job_id=applied['id'],continuum_type=kind,reverse_direction=reverse,generation={'step_count':3}))
            assert generated['state']=='succeeded',generated
            raw=outputs(runtime,generated);assert len(raw)==6
            parts=[]
            for i in range(3):
                fs,a=wavfile.read(io.BytesIO(raw[f'step{i+1:02d}.wav']));assert fs==11025 and a.dtype==np.int16;parts.append(a)
            fs,combined=wavfile.read(io.BytesIO(raw['combined_steps.wav']));np.testing.assert_array_equal(combined,np.concatenate(parts))
            assert evidence['cleaned'];all_ids.extend(f['id'] for f in generated['result_manifest']['files'])
    assert len(all_ids)==len(set(all_ids))

def test_identity_stale_and_missing_backend(runtime):
    request=body(runtime);job=submit(runtime[0],'local',request);assert submit(runtime[0],'local',request)['id']==job['id']
    with pytest.raises(JobError):submit(runtime[0],'other',request)
    request['analysis']={'lpc_order':18}
    with pytest.raises(JobError,match='idempotency_conflict'):submit(runtime[0],'local',request)
    runtime[0].cancel('local',job['id'])
    with pytest.raises(JobError,match='analysis_unavailable'):submit(runtime[0],'local',body(runtime,'generate',analysis_job_id=job['id']))

def test_cancel_and_partial_completed_group(runtime):
    analysis,_=run(runtime,body(runtime));assert analysis['state']=='succeeded'
    completed,_=run(runtime,body(runtime,'generate',analysis_job_id=analysis['id']));assert completed['state']=='succeeded'
    prior=outputs(runtime,completed)
    stop=threading.Event();cancelled,ev=run(runtime,body(runtime,'generate',analysis_job_id=analysis['id']),stop,on_started=lambda pid:stop.set())
    assert cancelled['state']=='cancelled' and cancelled['result_manifest'] is None and ev['cleaned']
    assert outputs(runtime,completed)==prior
    retry=runtime[0].retry('local',cancelled['id'],uuid4().hex);assert retry['state']=='queued'

@pytest.mark.parametrize('change',[{'analysis':{'frame_shift':128}},{'analysis':{'preemphasis':float('nan')}},{'generation':{'step_count':51}}])
def test_invalid_parameters(runtime,change):
    with pytest.raises(ValueError):M07Request.model_validate(body(runtime,**change))

def test_native_reaper_actual_and_missing_explicit(runtime):
    runtime[1].reaper_binary=None
    with pytest.raises(JobError,match='m07_reaper_unavailable'):submit(runtime[0],'local',body(runtime,analysis={'f0_backend':'reaper'}))
    runtime[1].reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe'
    # A pure sine is correctly unvoiced for this native estimator; use rich periodic input.
    for i,hz in enumerate((120,180)):
        t=np.arange(16000)/16000
        audio=sum(np.sin(2*np.pi*hz*k*t)/k for k in range(1,30))*.15
        stream=io.BytesIO();wavfile.write(stream,16000,np.round(audio*32767).astype(np.int16))
        runtime[2][i]=runtime[1].import_input(stream.getvalue(),f'reaper{i}.wav','audio')
    result,ev=run(runtime,body(runtime,analysis={'f0_backend':'reaper'}))
    assert result['state']=='succeeded',json.dumps(result)
    assert json.loads(outputs(runtime,result)['m07.ptb.json'])['analysis']['f0_backend']=='reaper'
    assert ev['cleaned']

def test_timeout_and_work_process_failure(runtime,monkeypatch):
    import ptb_worker.m07_executor as module
    from dataclasses import replace
    monkeypatch.setattr(module,'LIMITS',replace(module.LIMITS,timeout_seconds=.001))
    result,ev=run(runtime,body(runtime));assert result['state']=='failed' and result['error_code']=='m07_timeout' and ev['cleaned'],result
    monkeypatch.undo()
    def terminate_owned(pid):
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.restype=ctypes.c_void_p
        handle=kernel.OpenProcess(1,False,pid)
        if not handle:raise OSError('owned child unavailable')
        try:
            kernel.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint];kernel.TerminateProcess(handle,97)
        finally:
            kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.CloseHandle(handle)
    result,ev=run(runtime,body(runtime),on_started=terminate_owned)
    assert result['state']=='failed' and result['result_manifest'] is None and ev['cleaned']
