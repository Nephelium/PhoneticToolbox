"""M07 admission and immutable analysis dependencies, using shared transactions."""
import hashlib
import json
import sys
from uuid import uuid4
from .store import JobError,LOCAL_PROJECT,canonical,core_version,adapter_version,public
from ptb_api.m07_models import M07Request

def capability(store):
    from importlib.metadata import version,PackageNotFoundError
    if sys.platform!='win32' or getattr(store,'files',None) is None:return False
    try:return all(version(k)==v for k,v in {'numpy':'2.2.6','scipy':'1.16.3','praat-parselmouth':'0.4.7'}.items())
    except PackageNotFoundError:return False

def submit(store,owner,body,*,retry_of=None):
    body=M07Request.model_validate(body)
    if not capability(store):raise JobError('m07_platform_unverified',503)
    if body.analysis.f0_backend=='reaper':
        from pathlib import Path
        from .native.reaper import REAPER_SHA256
        binary=Path(getattr(store.files,'reaper_binary',None) or '')
        if not binary.is_file() or binary.stat().st_size>16_000_000 or hashlib.sha256(binary.read_bytes()).hexdigest()!=REAPER_SHA256:raise JobError('m07_reaper_unavailable',503)
    request=body.model_dump(exclude={'idempotency_key'})|{'retry_of':retry_of}
    digest=hashlib.sha256(canonical(request).encode()).hexdigest()
    with store.files.batch_transaction() as tx:
        old=tx.execute('SELECT * FROM {jobs} WHERE owner_id=? AND idempotency_key=?',(owner,body.idempotency_key)).fetchone()
        if old:
            if old['request_hash'].strip()!=digest:raise JobError('idempotency_conflict')
            return public(dict(old))
        if store.postgres:
            if not tx.execute('SELECT 1 FROM ptb_accounts.projects p JOIN ptb_accounts.users u ON p.owner_id=u.id WHERE p.id=? AND p.owner_id=? AND u.active',(body.project_id,owner)).fetchone():raise JobError('project_not_found',404)
        elif owner!='local' or body.project_id!=LOCAL_PROJECT:raise JobError('project_not_found',404)
        if store.capacity_used(tx,owner)>=1000:raise JobError('job_limit_reached')
        now=tx.now();inputs=[];refs={}
        for role in ('source','target'):
            ref=getattr(body,role).model_dump();refs[role]=ref
            item=store.files.validate_batch_input(tx,owner,body.project_id,{'audio':ref},now)[0]
            if item['size_bytes']>8_000_000 or not item['name'].lower().endswith('.wav'):raise JobError('m07_input_budget',413)
            inputs.append(dict(item,role=role))
        if body.analysis_job_id:
            old=store._row(tx,body.analysis_job_id,owner)
            if not old or str(old['project_id'])!=body.project_id or old['state']!='succeeded':raise JobError('m07_analysis_unavailable',409)
            snapshot=json.loads(old['snapshot']);previous=snapshot.get('request',{})
            if snapshot['operation']!='phonation_synthesis' or previous.get('action') not in ('analyze','apply') or any(previous.get(k)!=request[k] for k in ('source','target','analysis')):raise JobError('m07_analysis_stale',409)
            artifacts=json.loads(old['result_manifest'])['files'];saved=next((f for f in artifacts if f['name']=='analysis.m07.json'),None)
            if saved is None:raise JobError('m07_analysis_unavailable',409)
            ref=dict(asset_id=saved['id'],sha256=saved['sha256']);refs['analysis']=ref
            inputs.extend(store.files.validate_batch_input(tx,owner,body.project_id,{'m07_analysis':ref},now))
            inputs[-1]['role']='analysis'
        # Max-size estimate is conservative before decode; child refines from sample counts.
        estimate=dict(groups=1,steps=body.generation.step_count if body.action=='generate' else 0,max_samples_per_input=110250,max_pcm_bytes=110250*body.generation.step_count*4+4096,process_bytes=1_000_000_000)
        snapshot=dict(operation='phonation_synthesis',schema_version='m07/1',project_id=body.project_id,retry_of=retry_of,request=request,
            core_version=core_version,adapter_version=adapter_version,source_ids=['SRC-ZAIWA','REF-ZAIWA','SRC-PRAAT','SRC-REAPER'],input_refs=refs,input_assets=inputs,
            config=dict(inputs=list(dict.fromkeys(i['id'] for i in inputs)),analysis=request,estimate=estimate,max_output_bytes=128_000_000))
        job_id=str(uuid4())
        tx.execute("INSERT INTO {jobs}(id,owner_id,project_id,idempotency_key,request_hash,snapshot,state,deadline,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?,?)",(job_id,owner,body.project_id,body.idempotency_key,digest,canonical(snapshot),now+300,now,now))
        store.files.link_batch_inputs(tx,job_id,dict(owner_id=owner,project_id=body.project_id),list({i['id']:i for i in inputs}.values()),now)
        job=store._row(tx,job_id);store._event(tx,job,'queued',now);return public(job)
