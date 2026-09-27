"""Explicit M08 result IDs, immutable provenance and transactional display names.

Uses existing storage locks and job JSON manifests. Does not change ProjectStorage,
quota policy, expiry policy or schema. Deletion is never inferred from a prefix.
"""
from contextlib import contextmanager
import hashlib
import json
import re
from pathlib import PurePath
from uuid import uuid4
from .store import JobError, canonical, public
from ptb_api.quota import StorageError


@contextmanager
def locked(store):
    files=store.files
    if store.postgres:
        with files.storage._locked() as conn, files.tx(conn) as tx: yield tx
    else:
        with files.locked(), store.transaction() as tx: yield tx


def rows(store, tx, owner, project):
    return [dict(r) for r in tx.execute("SELECT * FROM {jobs} WHERE owner_id=? AND project_id=? ORDER BY created_at,id",(owner,project)).fetchall()
            if json.loads(r['snapshot'])['operation']=='pitch_manipulation']


def readable(store, tx, owner, item):
    if store.postgres:
        a=store.files.storage._readable(tx.conn,owner,item['id'])
        path=store.files.storage._path(item['id'])
    else:
        a=store.files._asset(item['id']); store.files._readable(tx,a)
        path=store.files._path(item['id'])
    if a['sha256']!=item['sha256'] or a['size_bytes']!=item['size_bytes']:
        raise JobError('input_unavailable',410)
    return path


def raw_file(store,tx,owner,item):
    raw=readable(store,tx,owner,item).read_bytes()
    if len(raw)>64_000_000 or hashlib.sha256(raw).hexdigest()!=item['sha256']:
        raise JobError('input_unavailable',410)
    return raw


def matching(row, source):
    snapshot=json.loads(row['snapshot'])
    ref=snapshot.get('source_ref',snapshot['input_refs']['audio'])
    return ref==source


def metadata(store,tx,owner,row):
    manifest=json.loads(row['result_manifest'])
    item=next(f for f in manifest['files'] if f['name']=='m08.ptb.json')
    return json.loads(raw_file(store,tx,owner,item))


def descriptors(store,tx,owner,row):
    if row['state']!='succeeded':return []
    manifest=json.loads(row['result_manifest']); meta=metadata(store,tx,owner,row)
    snapshot=json.loads(row['snapshot']); source=snapshot.get('source_ref',snapshot['input_refs']['audio'])
    result=[]
    for data in meta.get('results',[]):
        file=next(f for f in manifest['files'] if f['name']==data['name'])
        if file['id'] in manifest.get('deleted',[]):continue
        readable(store,tx,owner,file)
        result.append(dict(data,id=file['id'],name=manifest.get('aliases',{}).get(file['id'],file['name']),
                           source_id=source['asset_id'],job_id=row['id'],sha256=file['sha256'],
                           saved=snapshot.get('saved_copy',False) or snapshot['config']['analysis']['action'] in ('transform','linear')))
    return result


def listing(store,owner,project,source=None):
    with locked(store) as tx:
        out=[]
        for row in rows(store,tx,owner,project):
            if source and not matching(row,source):continue
            snapshot=json.loads(row['snapshot'])
            if snapshot['config']['analysis']['action']=='preview':continue
            try:results=descriptors(store,tx,owner,row)
            except (StorageError,JobError,OSError):results=[]
            ref=snapshot.get('source_ref',snapshot['input_refs']['audio'])
            out.append(dict(id=row['id'],source_id=ref['asset_id'],state=row['state'],results=results,error=row['error_code']))
        return out


def selected(store,tx,owner,body):
    if len(set(body.ids))!=len(body.ids):raise JobError('m08_duplicate_id',422)
    found={}
    for row in rows(store,tx,owner,body.project_id):
        if row['state']!='succeeded' or not matching(row,body.source.model_dump()):continue
        manifest=json.loads(row['result_manifest'])
        for item in manifest['files']:
            if item['id'] in body.ids and item['name'].endswith('.wav') and item['id'] not in manifest.get('deleted',[]):
                readable(store,tx,owner,item); found[item['id']]=(row,manifest,item)
    if len(found)!=len(body.ids):raise JobError('m08_result_not_found',404)
    return found


def manage(store,owner,body,action):
    with locked(store) as tx:
        found=selected(store,tx,owner,body)
        if action=='rename':
            names=body.names
            if len(names)!=len(body.ids) or len({n.casefold() for n in names})!=len(names):raise JobError('m08_name_conflict',409)
            if any(not n or len(n)>180 or n.startswith('.') or not n.lower().endswith('.wav') or any(ord(c)<32 or c in '/\\:<>"|?*' for c in n) for n in names):
                raise JobError('m08_invalid_name',422)
            used=set()
            for row in rows(store,tx,owner,body.project_id):
                if not row['result_manifest']:continue
                m=json.loads(row['result_manifest'])
                used.update(m.get('aliases',{}).get(f['id'],f['name']).casefold() for f in m['files'] if f['id'] not in body.ids and f['id'] not in m.get('deleted',[]))
            if used.intersection(n.casefold() for n in names):raise JobError('m08_name_conflict',409)
        removed=[]; failed=[]
        # Reuse one mutable manifest for multiple IDs in the same job.
        manifests={r['id']:m for r,m,_ in found.values()}
        for index,result_id in enumerate(body.ids):
            row,_,item=found[result_id]; manifest=manifests[row['id']]
            if action=='rename':manifest.setdefault('aliases',{})[result_id]=body.names[index]
            else:
                try:
                    if store.postgres:
                        a=store.files.storage._readable(tx.conn,owner,result_id)
                        state=store.files.storage._delete(tx.conn,a,notify_jobs=False)
                        if state['state']!='deleted':raise JobError('m08_delete_failed')
                    else:store.files._delete(store.files._asset(result_id))
                    manifest.setdefault('deleted',[]).append(result_id);removed.append(result_id)
                except (OSError,StorageError,JobError):failed.append(dict(id=result_id,error='m08_delete_failed'))
        for job_id,manifest in manifests.items():
            tx.execute('UPDATE {jobs} SET result_manifest=? WHERE id=?',(canonical(manifest),job_id))
        return dict(removed=removed,failed=failed) if action=='remove' else dict(renamed=body.ids)


def save_copy(store,owner,body):
    """A copy is a persistent fenced job. Original PCM and expiry remain unchanged."""
    if len(body.ids)!=1:raise JobError('m08_one_result_required',422)
    with locked(store) as tx:
        row,manifest,item=selected(store,tx,owner,body)[body.ids[0]]
        meta=metadata(store,tx,owner,row)
        result=next(r for r in meta['results'] if r['name']==item['name'])
        original=json.loads(row['snapshot']); source=body.source.model_dump()
        names=[]
        for other in rows(store,tx,owner,body.project_id):
            snap=json.loads(other['snapshot'])
            if snap.get('copy_name'):names.append(snap['copy_name'])
            if other['result_manifest'] and (snap.get('saved_copy') or snap['config']['analysis']['action'] in ('transform','linear')):
                m=json.loads(other['result_manifest'])
                names.extend(m.get('aliases',{}).get(f['id'],f['name']) for f in m['files'])
        stem=PurePath(original.get('source_name',original['input_assets'][0]['name'])).stem
        prefix=f"{stem}_{result['start']:.2f}_{result['end']:.2f}_modified"
        numbers=[int(m.group(1)) for n in names if n.casefold().startswith(prefix.casefold()+'_') and (m:=re.search(r'_(\d+)\.wav$',n,re.I))]
        name=f'{prefix}_{max(numbers,default=0)+1}.wav'
        now=tx.now()
        if item['expires_at'] is not None and item['expires_at']<=now+300:raise JobError('input_unavailable',410)
        if store.capacity_used(tx,owner)>=1000:raise JobError('job_limit_reached')
        job_id=str(uuid4()); key='m08-save-'+job_id
        src=dict(id=item['id'],sha256=item['sha256'],name=item['name'],size_bytes=item['size_bytes'],expires_at=item['expires_at'],role='audio')
        snapshot=dict(original,saved_copy=True,copy_name=name,copy_result=result,source_ref=source,source_name=stem+'.wav',
                      input_assets=[src],input_refs={'audio':dict(asset_id=item['id'],sha256=item['sha256'])})
        snapshot['config']=dict(inputs=[item['id']],analysis=result['config'],max_output_bytes=200_000_000)
        tx.execute("INSERT INTO {jobs}(id,owner_id,project_id,idempotency_key,request_hash,snapshot,state,deadline,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?,?)",
                   (job_id,owner,body.project_id,key,hashlib.sha256(key.encode()).hexdigest(),canonical(snapshot),now+300,now,now))
        store.files.link_batch_inputs(tx,job_id,dict(owner_id=owner,project_id=body.project_id),[src],now)
        job=store._row(tx,job_id);store._event(tx,job,'queued',now)
        return public(job)
