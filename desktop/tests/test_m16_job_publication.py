"""Deterministic owner/result races; no device, sleeps or spawned processes."""
import copy
import time
import pytest
import numpy as np
from ptb_desktop.recording.service import RecordingService
from ptb_desktop.recording.storage import Project,save_pcm,atomic_json,read_json


class Owner:
    alive=True
    on_exit=None
    def is_alive(self):
        if not self.alive and self.on_exit:
            self.on_exit();self.on_exit=None
        return self.alive


def make_job(tmp_path,kind='gain',state='complete'):
    service=RecordingService();service.project=Project(tmp_path/'project',True)
    span=save_pcm(service.project.root,'takes/take/raw/0.f32',np.zeros((16,2),np.float32))
    item={'id':'take','status':'complete','spans':[span],'config':{'sample_rate':48000,'channels':2,'roles':['microphone','egg']}}
    service.commit_capture(copy.deepcopy(item))
    take=service.take('take');result=tmp_path/'result.json';owner=Owner()
    payload={'state':state,'frames':16,'spans':[span],'metadata':{'test':'publication'},'progress':1}
    if kind=='export':payload={'state':state,'success_count':1,'total':1,'items':[]}
    if state=='failed':payload['error']='injected failure'
    atomic_json(result,payload)
    service.job={'kind':kind,'owner':owner,'result':result,'cancel':tmp_path/'cancel','take_id':'take','source_version':take['versions'][take['head']]['id'],'started':time.monotonic(),'cancelled':False}
    return service,owner,result,payload


@pytest.mark.parametrize('kind',['gain','denoise'])
def test_complete_result_waits_for_owner_exit_and_project_commit(tmp_path,kind):
    service,owner,result,payload=make_job(tmp_path,kind)
    try:
        for _ in range(2):
            assert service.poll_job()['state']=='running'
            assert service.job is not None and len(service.take('take')['versions'])==1
            with pytest.raises(ValueError,match='后台处理'):service.require(idle=True)
        owner.alive=False
        terminal=service.poll_job()
        assert terminal['state']=='complete' and terminal['project']['takes'][0]['head']==1
        assert service.take('take')['versions'][1]['kind']==('denoised' if kind=='denoise' else 'gain')
        assert read_json(service.project.root/'project.json')['takes'][0]['head']==1
        assert service.job is None and service.poll_job() is None
        assert len(service.take('take')['versions'])==2
    finally:service.project.close()


@pytest.mark.parametrize('kind,state',[('export','complete'),('export','failed'),('gain','failed'),('gain','cancelled')])
def test_terminal_status_and_busy_gate_wait_for_owner_exit(tmp_path,kind,state):
    service,owner,result,payload=make_job(tmp_path,kind,state)
    try:
        assert service.poll_job()['state']==('cancelling' if state=='cancelled' else 'running')
        with pytest.raises(ValueError,match='后台处理'):service.require(idle=True)
        owner.alive=False
        terminal=service.poll_job()
        assert terminal['state']==state and service.job is None
        assert len(service.take('take')['versions'])==1
        if kind=='export' and state=='complete':assert terminal['success_count']==1
        if state=='failed':assert terminal['error']=='injected failure'
        service.require(idle=True)
    finally:service.project.close()


def test_cancel_after_result_before_exit_never_publishes_version(tmp_path):
    service,owner,result,payload=make_job(tmp_path)
    try:
        service.dispatch({'op':'job_cancel'})
        assert service.poll_job()['state']=='cancelling'
        assert service.job is not None
        owner.alive=False
        assert service.poll_job()['state']=='cancelled'
        assert service.job is None and len(service.take('take')['versions'])==1
    finally:service.project.close()


def test_result_is_read_after_observing_exit(tmp_path):
    service,owner,result,payload=make_job(tmp_path)
    try:
        atomic_json(result,{'state':'running','progress':.5})
        owner.alive=False;owner.on_exit=lambda:atomic_json(result,payload)
        assert service.poll_job()['state']=='complete'
        assert service.take('take')['head']==1 and service.job is None
    finally:service.project.close()


def test_commit_failure_keeps_job_owned_for_retry(tmp_path,monkeypatch):
    service,owner,result,payload=make_job(tmp_path)
    try:
        owner.alive=False;commit=service.project.commit
        def fail_commit(data):raise OSError('injected manifest failure')
        monkeypatch.setattr(service.project,'commit',fail_commit)
        with pytest.raises(OSError,match='manifest failure'):service.poll_job()
        assert service.job is not None and service.take('take')['head']==0
        assert read_json(service.project.root/'project.json')['takes'][0]['head']==0
        monkeypatch.setattr(service.project,'commit',commit)
        assert service.poll_job()['state']=='complete'
        assert service.job is None and service.take('take')['head']==1
    finally:service.project.close()
