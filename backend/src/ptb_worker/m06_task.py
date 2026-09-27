"""M06 admission: existing project/owner transaction and immutable input refs."""
import hashlib
import secrets
import sys
from uuid import uuid4
from .store import JobError,LOCAL_PROJECT,canonical,core_version,adapter_version,public
from ptb_api.m06_models import M06Request


def capability(store):
    from importlib.metadata import version,PackageNotFoundError
    from importlib.util import find_spec
    if sys.platform!='win32' or getattr(store,'files',None) is None:return False
    if not all(callable(getattr(store.files,k,None)) for k in ('batch_transaction','read_input','output','complete')):return False
    try:
        return all(version(k)==v for k,v in {'numpy':'2.2.6','scipy':'1.16.3','praat-parselmouth':'0.4.7','soundfile':'0.13.1','pandas':'2.3.3'}.items()) and find_spec('ptb_worker.m06_child') is not None
    except (PackageNotFoundError,ImportError):return False


def submit(store,owner,body,*,retry_of=None):
    body=M06Request.model_validate(body)
    if not capability(store):raise JobError('m06_platform_unverified',503)
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
        now=tx.now();refs={'table':body.parameters.model_dump()} | ({'audio':body.audio.model_dump()} if body.audio else {})
        sources=store.files.validate_batch_input(tx,owner,body.project_id,refs,now) if refs else []
        if any(v['size_bytes']>8_000_000 for v in sources):raise JobError('m06_input_budget',413)
        snapshot=dict(operation='speech_synthesis',schema_version='m06/1',project_id=body.project_id,retry_of=retry_of,
            core_version=core_version,adapter_version=adapter_version,source_ids=['SRC-TDKLATT','REF-KLATT','SRC-PRAAT'],
            input_refs=refs,input_assets=sources,config=dict(inputs=[i['id'] for i in sources],
                action=body.action,seed=secrets.randbits(32),max_output_bytes=64_000_000))
        job_id=str(uuid4())
        tx.execute("INSERT INTO {jobs}(id,owner_id,project_id,idempotency_key,request_hash,snapshot,state,deadline,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?,?)",(job_id,owner,body.project_id,body.idempotency_key,sha,canonical(snapshot),now+300,now,now))
        if sources:store.files.link_batch_inputs(tx,job_id,dict(owner_id=owner,project_id=body.project_id),sources,now)
        job=store._row(tx,job_id);store._event(tx,job,'queued',now);return public(job)
