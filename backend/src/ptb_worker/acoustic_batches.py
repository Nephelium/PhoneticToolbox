"""005 durable ordered batches; P06 child tasks remain the only execution states."""
import hashlib
import json
from uuid import uuid4
from .batch_policy import BatchRequest,freeze_request,summarize,check_capacity
from .store import JobError,canonical,core_version,adapter_version,LOCAL_PROJECT
from .policy import cancel_state

OPERATIONS=('acoustic_analysis','textgrid_segment')


class AcousticBatches:
    def __init__(self,jobs,resources):
        self.jobs,self.resources=jobs,resources
        self.prefix='ptb_jobs.' if jobs.postgres else ''
        with jobs.transaction(write=False) as tx:
            if [dict(r) for r in tx.execute(f'SELECT version FROM {self.prefix}acoustic_batch_version')]!=[{'version':1}]:
                raise RuntimeError('M01 schema version mismatch')
        jobs.batches=self

    def table(self,name):return self.prefix+'acoustic_batch_'+name if name!='batches' else self.prefix+'acoustic_batches'

    def _row(self,tx,batch_id,owner):
        row=tx.execute(f'SELECT * FROM {self.table("batches")} WHERE id=? AND owner_id=?',(batch_id,owner)).fetchone()
        if not row:raise JobError('batch_not_found',404)
        return dict(row)

    def _items(self,tx,row):
        return [dict(r) for r in tx.execute(f'''SELECT i.*,j.state,j.progress,j.error_code,j.snapshot FROM {self.table('items')} i
            LEFT JOIN {{jobs}} j ON j.id=i.child_job_id WHERE i.batch_id=? ORDER BY i.ordinal''',(row['id'],))]

    def _view(self,tx,row):
        items=self._items(tx,row)
        result=summarize(row['id'],[dict(index=i['ordinal'],audio_asset_id=str(i['audio_asset_id']),
            job_id=i['child_job_id'],state=i['state'] or 'not_started',error_code=i['error_code']) for i in items],
            cancel_requested=bool(row['cancel_requested'])).model_dump()
        names=json.loads(row['config_snapshot']).get('audio_names') or [next((a['name'] for a in json.loads(i['snapshot'])['input_assets'] if a['role']=='audio'),f'Audio {i["ordinal"]+1}') if i['snapshot'] else f'Audio {i["ordinal"]+1}' for i in items]
        return dict(id=row['id'],project_id=str(row['project_id']),operation=row['operation'],audio_names=names,
                    created_at=row['created_at'],updated_at=row['updated_at'],cancel_requested=bool(row['cancel_requested']),
                    request_sha256=row['request_hash'].strip(),summary=result)

    def submit(self,owner,body):
        body=BatchRequest.model_validate(body);frozen=freeze_request(body)
        common=json.loads(frozen.serialized);inputs=common.pop('inputs')
        with self.resources.batch_transaction() as tx:
            old=tx.execute(f'SELECT * FROM {self.table("batches")} WHERE owner_id=? AND idempotency_key=?',(owner,body.idempotency_key)).fetchone()
            if old:
                if old['request_hash'].strip()!=frozen.sha256:raise JobError('idempotency_conflict')
                return self._view(tx,dict(old))
            if self.jobs.postgres:
                found=tx.execute('SELECT 1 FROM ptb_accounts.projects p JOIN ptb_accounts.users u ON p.owner_id=u.id WHERE p.id=? AND p.owner_id=? AND u.active',(body.project_id,owner)).fetchone()
                if not found:raise JobError('project_not_found',404)
            elif owner!='local' or body.project_id!=LOCAL_PROJECT:raise JobError('project_not_found',404)
            try:check_capacity(self.jobs.capacity_used(tx,owner),0,len(inputs))
            except ValueError:raise JobError('job_limit_reached',409) from None
            now=tx.now()
            # Admission validates all sources; creation and every use revalidate them.
            common['audio_names']=[]
            for item in inputs:
                resolved=self.resources.validate_batch_input(tx,owner,body.project_id,item,now)
                common['audio_names'].append(next(a['name'] for a in resolved if a['role']=='audio'))
            batch_id=str(uuid4())
            tx.execute(f'''INSERT INTO {self.table('batches')}(id,owner_id,project_id,idempotency_key,request_hash,
                operation,config_snapshot,total,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)''',
                (batch_id,owner,body.project_id,body.idempotency_key,frozen.sha256,body.operation,canonical(common),len(inputs),now,now))
            for index,item in enumerate(inputs):
                tx.execute(f'''INSERT INTO {self.table('items')}(batch_id,ordinal,owner_id,project_id,audio_asset_id,input_snapshot)
                    VALUES(?,?,?,?,?,?)''',(batch_id,index,owner,body.project_id,item['audio']['asset_id'],canonical(item)))
            row=self._row(tx,batch_id,owner);self._advance_one(tx,row,now)
            return self._view(tx,row)

    def _advance_one(self,tx,row,now):
        items=self._items(tx,row)
        if row['closed_at'] is not None:return
        if any(i['state'] in ('queued','running','cancel_requested') for i in items):return
        pending=next((i for i in items if i['child_job_id'] is None),None)
        if row['cancel_requested'] or pending is None:
            tx.execute(f'UPDATE {self.table("batches")} SET closed_at=?,updated_at=? WHERE id=?',(now,now,row['id']))
            return
        common=json.loads(row['config_snapshot']);inputs=json.loads(pending['input_snapshot'])
        key=f'm01-{row["id"]}-{pending["ordinal"]}-{pending["attempt"]}'
        source_error=None
        try:resolved=self.resources.validate_batch_input(tx,str(row['owner_id']),str(row['project_id']),inputs,now)
        except (JobError,ValueError) as exc:
            resolved=[];source_error='input_unavailable'
        # Resource implementations normalize source failures into JobError above.
        snapshot=dict(operation=row['operation'],project_id=str(row['project_id']),retry_of=None,
            core_version=core_version,adapter_version=adapter_version,source_ids=['SRC-PRAAT','SRC-REAPER'],
            batch_id=row['id'],batch_index=pending['ordinal'],batch_sha256=row['request_hash'].strip(),
            input_refs=inputs,input_assets=resolved,layer=common['layer'],
            config={'inputs':[x['id'] for x in resolved],'analysis':common['config'],'max_output_bytes':192_000_000})
        encoded=canonical(snapshot)
        if len(encoded.encode())>16384:raise JobError('snapshot_too_large',413)
        job_id=str(uuid4());state='failed' if source_error else 'queued'
        tx.execute('''INSERT INTO {jobs}(id,owner_id,project_id,idempotency_key,request_hash,snapshot,state,
            deadline,created_at,updated_at,error_code) VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
            (job_id,row['owner_id'],row['project_id'],key,hashlib.sha256(encoded.encode()).hexdigest(),encoded,state,now+300,now,now,source_error))
        self.resources.link_batch_inputs(tx,job_id,row,resolved,now)
        tx.execute(f'UPDATE {self.table("items")} SET child_job_id=? WHERE batch_id=? AND ordinal=?',(job_id,row['id'],pending['ordinal']))
        tx.execute(f'UPDATE {self.table("batches")} SET updated_at=? WHERE id=?',(now,row['id']))
        job=self.jobs._row(tx,job_id);self.jobs._event(tx,job,source_error or 'queued',now)

    def advance(self,tx,now):
        rows=tx.execute(f'SELECT * FROM {self.table("batches")} WHERE closed_at IS NULL ORDER BY created_at,id').fetchall()
        for row in rows:self._advance_one(tx,dict(row),now)

    def get(self,owner,batch_id):
        with self.resources.batch_transaction() as tx:
            self.jobs._recover(tx,tx.now());row=self._row(tx,batch_id,owner)
            self._advance_one(tx,row,tx.now());return self._view(tx,self._row(tx,batch_id,owner))

    def list(self,owner,project_id):
        with self.jobs.transaction(write=False) as tx:
            return [self._view(tx,dict(r)) for r in tx.execute(f'SELECT * FROM {self.table("batches")} WHERE owner_id=? AND project_id=? ORDER BY created_at DESC,id DESC LIMIT 100',(owner,project_id))]

    def cancel(self,owner,batch_id):
        with self.resources.batch_transaction() as tx:
            now=tx.now();self.jobs._recover(tx,now);row=self._row(tx,batch_id,owner)
            if row['closed_at'] is not None:return self._view(tx,row)
            tx.execute(f'UPDATE {self.table("batches")} SET cancel_requested=?,updated_at=? WHERE id=?',(True,now,batch_id))
            for item in self._items(tx,row):
                if item['state'] in ('queued','running'):
                    job=self.jobs._row(tx,item['child_job_id']);state=cancel_state(job['state'])
                    tx.execute('UPDATE {jobs} SET state=? WHERE id=?',(state,job['id']));job['state']=state
                    self.jobs._event(tx,job,state,now)
            row=self._row(tx,batch_id,owner);self._advance_one(tx,row,now)
            return self._view(tx,self._row(tx,batch_id,owner))

    def parent_result(self,owner,project,sha):
        with self.resources.batch_transaction() as tx:
            rows=tx.execute("SELECT snapshot,result_manifest FROM {jobs} WHERE owner_id=? AND project_id=? AND state='succeeded' ORDER BY created_at DESC,id DESC LIMIT 1000",(owner,project))
            for row in rows:
                snap=json.loads(row['snapshot'])
                if snap['operation']!='acoustic_analysis' or snap['input_refs']['audio']['sha256']!=sha:continue
                file=next(f for f in json.loads(row['result_manifest'])['files'] if f['name'].endswith('.ptb.json'))
                reference=dict(asset_id=file['id'],sha256=file['sha256'])
                try:self.resources.validate_batch_input(tx,owner,project,{'parent_result':reference},tx.now())
                except JobError:continue
                return reference
        return None

    def retry(self,owner,job_id,key):
        from .store import public
        with self.resources.batch_transaction() as tx:
            old=self.jobs._row(tx,job_id,owner)
            if not old or old['state'] not in ('failed','interrupted','cancelled'):raise JobError('job_not_retryable')
            snap=json.loads(old['snapshot']);snap['retry_of']=job_id
            encoded=canonical(snap);digest=hashlib.sha256(encoded.encode()).hexdigest()
            existing=tx.execute('SELECT * FROM {jobs} WHERE owner_id=? AND idempotency_key=?',(owner,key)).fetchone()
            if existing:
                if existing['request_hash']!=digest:raise JobError('idempotency_conflict')
                return public(dict(existing))
            if self.jobs.capacity_used(tx,owner)>=1000:raise JobError('job_limit_reached')
            item=tx.execute(f'SELECT * FROM {self.table("items")} WHERE child_job_id=?',(job_id,)).fetchone()
            if not item or item['attempt']>=99:raise JobError('job_not_current')
            batch=self._row(tx,item['batch_id'],owner)
            if any(i['state'] in ('running','queued','cancel_requested') for i in self._items(tx,batch)):raise JobError('batch_busy')
            now=tx.now();sources=self.resources.validate_batch_input(tx,owner,str(old['project_id']),snap['input_refs'],now)
            new_id=str(uuid4())
            tx.execute("INSERT INTO {jobs}(id,owner_id,project_id,idempotency_key,request_hash,snapshot,state,deadline,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?,?)",
                (new_id,owner,old['project_id'],key,digest,encoded,now+300,now,now))
            self.resources.link_batch_inputs(tx,new_id,batch,sources,now)
            tx.execute(f'UPDATE {self.table("items")} SET child_job_id=?,attempt=attempt+1 WHERE batch_id=? AND ordinal=?',(new_id,batch['id'],item['ordinal']))
            tx.execute(f'UPDATE {self.table("batches")} SET closed_at=NULL,updated_at=? WHERE id=?',(now,batch['id']))
            job=self.jobs._row(tx,new_id);self.jobs._event(tx,job,'queued',now);return public(job)
