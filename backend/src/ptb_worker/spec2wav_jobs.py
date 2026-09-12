"""Generic P06 job admission using P07 input links, no batch DDL."""
import hashlib
from uuid import uuid4
from .store import JobError,LOCAL_PROJECT,canonical,core_version,adapter_version,public
from ptb_api.spec2wav_models import Spec2WavRequest


def submit(store,owner,body,*,retry_of=None):
    body=Spec2WavRequest.model_validate(body)
    if store.files is None:raise JobError('task_service_unavailable',503)
    request=body.model_dump(exclude={'idempotency_key'})|{'retry_of':retry_of}
    sha=hashlib.sha256(canonical(request).encode()).hexdigest()
    with store.files.batch_transaction() as tx:
        old=tx.execute('SELECT * FROM {jobs} WHERE owner_id=? AND idempotency_key=?',(owner,body.idempotency_key)).fetchone()
        if old:
            if old['request_hash'].strip()!=sha:raise JobError('idempotency_conflict')
            return public(dict(old))
        if store.postgres:
            if not tx.execute('SELECT 1 FROM ptb_accounts.projects p JOIN ptb_accounts.users u ON p.owner_id=u.id WHERE p.id=? AND p.owner_id=? AND u.active',(body.project_id,owner)).fetchone():raise JobError('project_not_found',404)
        elif owner!='local' or body.project_id!=LOCAL_PROJECT:raise JobError('project_not_found',404)
        if store.capacity_used(tx,owner)>=1000:raise JobError('job_limit_reached')
        now=tx.now();refs={'image':body.image.model_dump()}
        sources=store.files.validate_batch_input(tx,owner,body.project_id,refs,now)
        snapshot=dict(operation='spectrogram_to_audio',project_id=body.project_id,retry_of=retry_of,
            core_version=core_version,adapter_version=adapter_version,source_ids=['REF-GRIFFINLIM','M01-PY-NUMPY','M01-PY-SCIPY','PKG-OPENCV-CONTRIB-PYTHON','PKG-SOUNDFILE'],
            input_refs=refs,input_assets=sources,config=dict(inputs=[i['id'] for i in sources],analysis=body.config.model_dump(),max_output_bytes=96_000_000))
        job_id=str(uuid4())
        tx.execute("INSERT INTO {jobs}(id,owner_id,project_id,idempotency_key,request_hash,snapshot,state,deadline,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?,?)",
            (job_id,owner,body.project_id,body.idempotency_key,sha,canonical(snapshot),now+300,now,now))
        store.files.link_batch_inputs(tx,job_id,dict(owner_id=owner,project_id=body.project_id),sources,now)
        job=store._row(tx,job_id);store._event(tx,job,'queued',now)
        return public(job)
