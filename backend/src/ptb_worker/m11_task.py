"""M11 admission uses the current JobStore and P07 owner/input transactions."""
import hashlib
import os
from uuid import uuid4
from .store import JobError,LOCAL_PROJECT,canonical,core_version,adapter_version,public
from .mfa.components import safe_name
from .mfa.runtime import select,load_registry
from ptb_api.m11_models import M11Request


def catalog(*,local):
    data=load_registry()
    return dict(schema_version='m11/1',download_available=False,download_reason='m11_download_not_published',
        execution_available=local and os.name=='nt' and any(r.get('validated') for r in data['runtimes']),
        waiting_reason=None if local else 'm11_waiting_verified_node',
        runtimes=[{k:r.get(k) for k in ('id','version','platform','arch','validated','fingerprint','versions','installed_bytes','download_bytes','source','archive_sha256')} for r in data['runtimes']],
        models=[{k:m.get(k) for k in ('id','name','model_sha256','dictionary_sha256','validated_runtime','model_bytes','dictionary_bytes')} for m in data['models']])


def submit(store,owner,body,*,retry_of=None):
    body=M11Request.model_validate(body)
    if store.files is None:raise JobError('task_service_unavailable',503)
    # Model/runtime IDs are host allowlisted. No web-supplied model executable/path.
    data=load_registry()
    models=[m for m in data['models'] if m['id']==body.model_id]
    runtimes=[r for r in data['runtimes'] if r['id']==body.runtime_id]
    if not models or not runtimes:raise JobError('m11_component_not_registered',409)
    if not store.postgres:
        try:select(body.runtime_id,body.model_id,verify_content=False)
        except ValueError as exc:raise JobError(str(exc),409) from None
    names=[]
    for item in body.corpus:
        try:safe_name(item.name)
        except ValueError:raise JobError('m11_unsafe_path',422) from None
        if not item.name.lower().endswith('.wav') or len(item.name.encode('utf8'))>110:
            raise JobError('m11_input_name_budget',422)
        names.append(item.name.casefold())
    if len(set(names))!=len(names):raise JobError('m11_duplicate_input',422)
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
        now=tx.now();sources=[]
        for item in body.corpus:
            sources.extend(store.files.validate_batch_input(tx,owner,body.project_id,dict(audio=item.audio.model_dump(),transcript=item.transcript.model_dump()),now))
        if body.dictionary:
            sources.extend(store.files.validate_batch_input(tx,owner,body.project_id,dict(dictionary=body.dictionary.model_dump()),now))
        sources=list({s['id']:s for s in sources}.values())
        if sum(s['size_bytes'] for s in sources)>64_000_000:raise JobError('m11_input_budget',413)
        waiting=store.postgres or os.name!='nt'
        snapshot=dict(operation='mfa_alignment',schema_version='m11/1',project_id=body.project_id,retry_of=retry_of,
            core_version=core_version,adapter_version=adapter_version,source_ids=['SRC-MFA','REF-MFA'],
            input_assets=sources,request=request,waiting_reason='m11_waiting_verified_node' if waiting else None,
            execution_route='remote_pending' if waiting else 'desktop-local',
            model_sha256=models[0]['model_sha256'],dictionary_sha256=models[0]['dictionary_sha256'],
            runtime_fingerprint=runtimes[0]['fingerprint'],
            config=dict(inputs=[s['id'] for s in sources],max_output_bytes=64_000_000))
        # No remote claim is admitted until the existing remote/1 bridge is ready.
        deadline=min([now+259200]+[s['expires_at']-1 for s in sources if s.get('expires_at')]) if waiting else now+300
        job_id=str(uuid4())
        tx.execute("INSERT INTO {jobs}(id,owner_id,project_id,idempotency_key,request_hash,snapshot,state,deadline,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?,?)",(job_id,owner,body.project_id,body.idempotency_key,sha,canonical(snapshot),deadline,now,now))
        store.files.link_batch_inputs(tx,job_id,dict(owner_id=owner,project_id=body.project_id),sources,now)
        job=store._row(tx,job_id)
        store._event(tx,job,'m11_waiting_verified_node' if waiting else 'queued',now)
        return public(job)
