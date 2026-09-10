"""M01-F: public admission and binary limits; no database success simulation."""
import json
import subprocess
import sys
from uuid import uuid4
from fastapi.testclient import TestClient
from ptb_api.main import create_app
from ptb_worker.store import JobError,LOCAL_PROJECT


class Rejected:
    def __init__(self):self.batches=self;self.files=self;self.calls=[]
    def submit(self,owner,body):self.calls.append((owner,len(body.inputs)));raise JobError('input_unavailable',409)
    def import_input(self,raw,name,role):self.calls.append((len(raw),name,role));raise JobError('invalid_local_input',422)


def test_batch_admission_auth_large_bounded_json_and_no_owner_override():
    store=Rejected();origin='http://127.0.0.1:5177';token='x'*40
    headers={'Authorization':'Bearer '+token,'Origin':origin}
    inputs=[dict(audio=dict(asset_id=str(uuid4()),sha256='a'*64),textgrid=dict(asset_id=str(uuid4()),sha256='b'*64)) for _ in range(100)]
    body=dict(operation='textgrid_segment',project_id=LOCAL_PROJECT,idempotency_key='batch-test-01',inputs=inputs,layer='音节')
    with TestClient(create_app('local',job_store=store,local_token=token,local_origin=origin),base_url=origin) as c:
        route='/api/v1/jobs/batches/create'
        assert len(json.dumps(body))>16384
        assert c.post(route,json=body).status_code==403
        assert c.post(route,json=body,headers={'Authorization':'Bearer '+token}).status_code==403
        assert c.post(route,json=body|dict(owner_id='other'),headers=headers).status_code==422
        assert not store.calls
        assert c.post(route,json=body,headers=headers).status_code==409
        assert store.calls==[('local',100)]
        assert c.post(route,content=' '*1_000_001,headers=headers).status_code==413


def test_binary_local_input_checks_identity_before_reading_and_its_own_limit():
    store=Rejected();origin='http://127.0.0.1:5177';headers={'Authorization':'Bearer '+'x'*40,'Origin':origin}
    with TestClient(create_app('local',job_store=store,local_token='x'*40,local_origin=origin),base_url=origin) as c:
        path='/api/v1/jobs/local-inputs?role=textgrid&name=labels.TextGrid'
        assert c.post(path,content=b'x'*20001).status_code==403
        assert not store.calls
        assert c.post(path,content=b'x'*20001,headers=headers).status_code==422
        assert store.calls==[(20001,'labels.TextGrid','textgrid')]
        assert c.post(path,content=b'x'*2_000_001,headers=headers).status_code==413
        assert len(store.calls)==1


def test_orchestration_import_never_loads_scientific_dlls():
    result=subprocess.run([sys.executable,'-c',"import sys; import ptb_worker.acoustic_executor; from ptb_worker.native.reaper import collect_pipe; assert not ({'numpy','scipy','parselmouth','PyQt6'} & set(sys.modules))"],capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr.decode(errors='replace')
