"""M09 expanded stroke envelopes, auth, legacy hashes and preview revalidation."""
from pathlib import Path
import io
import sqlite3
import json
from uuid import uuid4
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient
from ptb_api.main import create_app
from ptb_api.spec2wav_models import Spec2WavRequest,Spec2WavConfig
from ptb_worker.store import SQLiteJobStore,LOCAL_PROJECT,JobError
from ptb_worker.local_acoustic_files import LocalAcousticFiles,initialize_local_files
from ptb_worker.spec2wav_jobs import submit
from ptb_worker.spec2wav_preview import preview_source,render
from ptb_worker.spectrogram_preview import PreviewError,preview_slot

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime(tmp_path):
    db=tmp_path/'tasks.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:source.backup(target)
    store=SQLiteJobStore(db);cache=tmp_path/'cache';cache.mkdir();initialize_local_files(cache);files=LocalAcousticFiles(store,cache)
    raw=io.BytesIO();sf.write(raw,np.sin(2*np.pi*400*np.arange(16000)/16000)*.1,16000,format='WAV',subtype='FLOAT')
    ref=files.import_input(raw.getvalue(),'synthetic.wav','audio')
    return store,files,ref,raw.getvalue()


def body(runtime,**config):return dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,image=runtime[2],config=dict(mode='audio_draw',**config))


def test_legacy_config_snapshot_keeps_exact_historical_fields():
    assert set(Spec2WavConfig().snapshot())=={'time_start','time_end','freq_start','freq_end','min_db','max_db','win_length_ms','n_iter','target_sr','seed','corners'}


@pytest.mark.parametrize('change',[{'project_id':str(uuid4())},{'image':{'asset_id':LOCAL_PROJECT,'sha256':'0'*64}}])
def test_foreign_or_missing_preview_input_is_rejected(runtime,change):
    with pytest.raises(JobError):preview_source(runtime[0],'local',Spec2WavRequest(**(body(runtime)|change)))


def test_other_owner_and_changed_source_after_compute_are_rejected(runtime,monkeypatch):
    store,files,ref,_=runtime;request=Spec2WavRequest(**body(runtime))
    with pytest.raises(JobError):preview_source(store,'other',request)
    def changed(*args):
        with files.locked() as state:
            state['assets'][ref['asset_id']]['sha256']='0'*64;files._save(state)
        return {'must_not_be_returned':True}
    monkeypatch.setattr('ptb_worker.spec2wav_preview.render',changed)
    with pytest.raises(JobError,match='input_unavailable'):preview_source(store,'local',request)


def test_large_stroke_request_is_bounded_and_authenticated(runtime):
    store=runtime[0];origin='http://127.0.0.1:5177';token='x'*40;headers={'Authorization':'Bearer '+token,'Origin':origin}
    request=body(runtime,strokes=[dict(color=0,size=3,opacity=.5,points=[dict(x=i/1500,y=.5) for i in range(1500)])])
    assert len(json.dumps(request))>16384
    with TestClient(create_app('local',job_store=store,local_token=token,local_origin=origin),base_url=origin) as c:
        for endpoint in ['create','preview']:
            assert c.post('/api/v1/jobs/spec2wav/'+endpoint,json=request).status_code==403
            assert c.post('/api/v1/jobs/spec2wav/'+endpoint,json=request|{'owner_id':'other'},headers=headers).status_code==422
            assert c.post('/api/v1/jobs/spec2wav/'+endpoint,content=' '*1_000_001,headers=headers).status_code==413
        response=c.post('/api/v1/jobs/spec2wav/create',json=request,headers=headers);assert response.status_code==201,response.text
        with store.transaction(write=False) as tx:
            encoded=store._row(tx,response.json()['id'])['snapshot'];snap=json.loads(encoded)
        assert len(encoded)<16384 and 'spectral_drawing' in snap['input_refs']
        from ptb_worker.spec2wav_drawing import expand
        drawing=runtime[1]._path(snap['input_refs']['spectral_drawing']['asset_id']).read_bytes()
        assert expand(snap['config']['analysis'],drawing)['strokes']==request['config']['strokes']
        assert c.post('/api/v1/jobs/spec2wav/create',json=request,headers=headers).json()['id']==response.json()['id']
        store.cancel('local',response.json()['id']);retry=store.retry('local',response.json()['id'],uuid4().hex)
        with store.transaction(write=False) as tx:retried=json.loads(store._row(tx,retry['id'])['snapshot'])
        assert retried['input_refs']==snap['input_refs']
        request['config']['strokes'][0]['opacity']=.6
        assert c.post('/api/v1/jobs/spec2wav/create',json=request,headers=headers).status_code==409


def test_audio_retry_keeps_role_and_immutable_strokes(runtime):
    store=runtime[0];request=body(runtime,strokes=[dict(points=[dict(x=.5,y=.5)])]);job=submit(store,'local',request)
    store.cancel('local',job['id']);retry=store.retry('local',job['id'],uuid4().hex)
    assert retry['retry_of']==job['id']
    with store.transaction(write=False) as tx:
        snap=json.loads(store._row(tx,retry['id'])['snapshot'])
    assert set(snap['input_refs'])=={'audio'} and snap['config']['analysis']['strokes'][0]['points']==request['config']['strokes'][0]['points']


def test_preview_timeout_and_busy_release_owned_resources(runtime):
    with pytest.raises(PreviewError,match='preview_timeout'):render(runtime[3],{'mode':'audio_draw'},timeout=.001)
    with preview_slot():
        with pytest.raises(PreviewError,match='preview_busy'):preview_source(runtime[0],'local',Spec2WavRequest(**body(runtime)))
    assert preview_source(runtime[0],'local',Spec2WavRequest(**body(runtime)))['samples']==16000
