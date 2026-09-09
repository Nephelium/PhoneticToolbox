from uuid import uuid4
from fastapi.testclient import TestClient
from ptb_api.main import create_app
from ptb_worker.store import JobError, LOCAL_PROJECT


class RejectedStore:
    """Boundary-only spy: never produces a successful task or database evidence."""
    def __init__(self):self.calls=[]
    def submit(self,owner,body):self.calls.append((owner,body));raise JobError('project_not_found',404)
    def get(self,owner,job_id):self.calls.append((owner,job_id));raise JobError('job_not_found',404)
    def events(self,owner,job_id,after):return self.get(owner,job_id)
    def cancel(self,owner,job_id):return self.get(owner,job_id)


def test_local_job_boundary_requires_token_origin_and_rejects_extra_owner():
    store=RejectedStore();origin='http://127.0.0.1:5177';token='x'*40
    app=create_app('local',job_store=store,local_token=token,local_origin=origin)
    with TestClient(app,base_url=origin) as c:
        body={'project_id':LOCAL_PROJECT,'idempotency_key':'probe_key_01'}
        headers={'Authorization':'Bearer '+token,'Origin':origin}
        assert c.post('/api/v1/jobs',json=body).status_code==403
        assert c.post('/api/v1/jobs',json=body,headers={'Authorization':'Bearer '+token}).status_code==403
        assert c.post('/api/v1/jobs',json=body|{'owner_id':'wrong'},headers=headers).status_code==422
        assert not store.calls
        assert c.post('/api/v1/jobs',json=body,headers=headers).status_code==404
        assert store.calls[0][0]=='local'
        assert c.get('/api/v1/jobs/'+str(uuid4()),headers=headers).headers['Cache-Control']=='no-store'
        assert c.post('/api/v1/jobs',content='x'*17000,headers=headers).status_code==413
