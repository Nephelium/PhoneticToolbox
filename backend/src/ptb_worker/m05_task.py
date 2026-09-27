"""M05 admission. Server/remote execution stays closed until separately admitted."""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4
from .store import JobError, LOCAL_PROJECT, canonical, core_version, adapter_version, public
from ptb_api.m05_models import M05Request

def catalog(local):
    runtime=os.environ.get('PTB_M05_PYTHON','')
    available=local and os.name=='nt' and bool(runtime) and Path(runtime).is_file()
    return dict(schema_version='m05/1',available=available,backend='legacy-facemesh/0.10.14',
                reason=None if available else 'm05_runtime_not_admitted',remote_available=False,
                input_bytes=128_000_000,output_bytes=512_000_000,process_bytes=2_147_483_648)

def submit(store,owner,body,*,retry_of=None):
    body=M05Request.model_validate(body)
    if store.files is None:raise JobError('task_service_unavailable',503)
    if not catalog(not store.postgres)['available']:raise JobError('m05_runtime_not_admitted',409)
    request=body.model_dump(exclude={'idempotency_key'})|dict(retry_of=retry_of)
    sha=hashlib.sha256(canonical(request).encode()).hexdigest()
    with store.files.batch_transaction() as tx:
        old=tx.execute('SELECT * FROM {jobs} WHERE owner_id=? AND idempotency_key=?',(owner,body.idempotency_key)).fetchone()
        if old:
            if old['request_hash'].strip()!=sha:raise JobError('idempotency_conflict')
            return public(dict(old))
        if owner!='local' or body.project_id!=LOCAL_PROJECT:raise JobError('project_not_found',404)
        if store.capacity_used(tx,owner)>=1000:raise JobError('job_limit_reached')
        now=tx.now();sources=store.files.validate_batch_input(tx,owner,body.project_id,dict(video=body.video.model_dump()),now)
        if sources[0]['size_bytes']>128_000_000:raise JobError('m05_input_budget',413)
        snapshot=dict(operation='lip_analysis',schema_version='m05/1',project_id=body.project_id,retry_of=retry_of,
            core_version=core_version,adapter_version=adapter_version,source_ids=['SRC-MEDIAPIPE'],input_assets=sources,
            request=request,execution_route='desktop-local',config=dict(inputs=[s['id'] for s in sources],max_output_bytes=512_000_000))
        key=str(uuid4())
        tx.execute("INSERT INTO {jobs}(id,owner_id,project_id,idempotency_key,request_hash,snapshot,state,deadline,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?,?)",(key,owner,body.project_id,body.idempotency_key,sha,canonical(snapshot),now+1800,now,now))
        job=store._row(tx,key);store._event(tx,job,'queued',now)
        return public(job)

def repeat(store,owner,key,config):
    with store.transaction(write=False) as tx:
        job=store._row(tx,key,owner)
        if not job or job['state']!='succeeded':raise JobError('job_not_found',404)
        snapshot=json.loads(job['snapshot'])
        if snapshot['operation']!='lip_analysis':raise JobError('job_not_found',404)
        original=snapshot['request']
    return submit(store,owner,dict(project_id=original['project_id'],idempotency_key=uuid4().hex,video=original['video'],config=config.model_dump()))
