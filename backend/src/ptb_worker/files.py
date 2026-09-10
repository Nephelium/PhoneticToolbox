"""P07 PG file jobs: bounded writer, durable references and atomic manifests.

Lock order is storage OS/session lock, then the short P06 scheduling transaction.
Disk writes retain the OS lock, but do not hold a scheduling transaction.
"""
from contextlib import contextmanager
import hashlib
import json
from uuid import uuid4

from ptb_api.job_models import FileManifest
from ptb_api.storage_models import UploadInput
from ptb_api.storage import public_asset
from ptb_api.quota import CHUNK_BYTES, QUOTA_BYTES, StorageError, expiry
from .store import Transaction, JobError, canonical, public, core_version, adapter_version
from .policy import fenced, cancel_state

FILE_OPERATIONS = ('storage_check','archive_zip','extract_zip')


class FilePipeline:
    def __init__(self, jobs, storage):
        if not jobs.postgres or jobs._dsn != storage._dsn:
            raise ValueError('File jobs require the same PostgreSQL and private storage')
        self.jobs, self.storage = jobs, storage
        with storage._locked() as conn:
            assert conn.execute('SELECT version FROM ptb_storage.job_files_version').fetchall() == [{'version':1}]
        jobs.files = self
        storage.files = self

    @contextmanager
    def tx(self, conn):
        with conn.transaction():
            conn.execute('SELECT pg_advisory_xact_lock(577606)')
            yield Transaction(conn, True)

    def submit(self, owner, body, *, retry_of=None):
        request = body.model_dump(exclude={'idempotency_key'}) | {'retry_of':retry_of}
        digest = hashlib.sha256(canonical(request).encode('utf-8')).hexdigest()
        with self.storage._locked() as conn, self.tx(conn) as tx:
            self.storage._writable(conn)
            old = tx.execute('SELECT * FROM {jobs} WHERE owner_id=? AND idempotency_key=?', (owner,body.idempotency_key)).fetchone()
            if old:
                if old['request_hash'] != digest: raise JobError('idempotency_conflict')
                return public(dict(old))
            if not conn.execute('SELECT 1 FROM ptb_accounts.projects p JOIN ptb_accounts.users u ON p.owner_id=u.id WHERE p.id=%s AND p.owner_id=%s AND u.active',(body.project_id,owner)).fetchone():
                raise JobError('project_not_found',404)
            if self.jobs.capacity_used(tx,owner) >= 1000:
                raise JobError('job_limit_reached')
            quota=conn.execute('SELECT used_bytes,reserved_bytes FROM ptb_storage.quota_accounts WHERE owner_id=%s',(owner,)).fetchone()
            if quota and quota['used_bytes']+quota['reserved_bytes']>=QUOTA_BYTES:
                raise StorageError('quota_exceeded',413)
            now = tx.now()
            inputs = []
            for asset_id in body.config.inputs:
                item = self.storage._readable(conn,owner,asset_id)
                if str(item['project_id']) != body.project_id:
                    raise StorageError('asset_not_found',404)
                # Conservative upper runtime bound; never promise input renewal.
                if item['expires_at'] <= now+300:
                    raise StorageError('input_lifetime_too_short',409)
                inputs.append(item)
            snapshot = request | {'core_version':core_version,'adapter_version':adapter_version,
                'source_ids':['P07-PYTHON-ZIP'],
                'input_assets':[{'id':str(a['id']),'sha256':a['sha256'],'expires_at':a['expires_at']} for a in inputs]}
            job_id = str(uuid4())
            tx.execute("INSERT INTO {jobs}(id,owner_id,project_id,idempotency_key,request_hash,snapshot,state,deadline,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?,?)",
                (job_id,owner,body.project_id,body.idempotency_key,digest,canonical(snapshot),now+300,now,now))
            for item in inputs:
                conn.execute("INSERT INTO ptb_storage.job_assets(job_id,asset_id,owner_id,project_id,role,generation,input_sha256,input_expires_at,created_at) VALUES(%s,%s,%s,%s,'input',0,%s,%s,%s)",
                    (job_id,item['id'],owner,body.project_id,item['sha256'],item['expires_at'],now))
            row = self.jobs._row(tx,job_id)
            self.jobs._event(tx,row,'queued',now)
            return public(row)

    def _fence(self, conn, identity):
        self.storage._writable(conn)
        with self.tx(conn) as tx:
            now = tx.now()
            row = self.jobs._row(tx,identity[0])
            if not row or not fenced(row,identity[1],identity[2],now):
                raise StorageError('stale_worker',409)
            if row['state'] != 'running':
                raise StorageError('cancelled',409)
            inputs = conn.execute("SELECT a.*,l.input_sha256,l.input_expires_at FROM ptb_storage.job_assets l JOIN ptb_storage.assets a ON a.id=l.asset_id WHERE l.job_id=%s AND l.role='input'",(row['id'],)).fetchall()
            for item in inputs:
                if item['state']!='ready' or min(item['expires_at'],item['input_expires_at']) <= now or item['sha256']!=item['input_sha256']:
                    raise StorageError('input_unavailable',410)
            order = json.loads(row['snapshot'])['config']['inputs']
            inputs.sort(key=lambda item:order.index(str(item['id'])))
            # Each bounded IO operation is also a lease heartbeat. Long work is
            # in a separate owned worker; no API request computes an archive.
            tx.execute('UPDATE {jobs} SET lease_until=? WHERE id=?',(now+self.jobs.lease_seconds,row['id']))
            return row, inputs

    def claim(self, worker_id):
        with self.storage._locked() as conn, self.tx(conn) as tx:
            self.storage._writable(conn)
            invalid = conn.execute("SELECT DISTINCT j.* FROM ptb_jobs.jobs j JOIN ptb_storage.job_assets l ON l.job_id=j.id JOIN ptb_storage.assets a ON a.id=l.asset_id WHERE j.state='queued' AND l.role='input' AND (a.state!='ready' OR a.expires_at<=j.deadline OR l.input_expires_at<=j.deadline OR a.sha256!=l.input_sha256)").fetchall()
            for value in invalid:
                row=dict(value);row['state']='failed'
                tx.execute("UPDATE {jobs} SET state='failed',error_code='input_unavailable' WHERE id=?",(row['id'],))
                self.jobs._event(tx,row,'input_unavailable',tx.now())
            return self.jobs._claim(tx,worker_id)

    def _outputs(self, conn, identity):
        return conn.execute("SELECT a.* FROM ptb_storage.job_assets l JOIN ptb_storage.assets a ON a.id=l.asset_id WHERE l.job_id=%s AND l.generation=%s AND l.role='output' ORDER BY a.created_at,a.id",(identity[0],identity[2])).fetchall()

    def _output(self, conn, identity, asset_id):
        result = next((a for a in self._outputs(conn,identity) if str(a['id'])==str(asset_id)), None)
        if result is None: raise StorageError('worker_owned_asset',403)
        return result

    def output(self, identity, name, kind, expected=None):
        if kind not in ('result','archive','temporary'):
            raise StorageError('invalid_output_kind',422)
        with self.storage._locked() as conn:
            job, inputs = self._fence(conn,identity)
            rows = self._outputs(conn,identity)
            if len(rows)>=self.output_limit(job): raise StorageError('output_count_exceeded',413)
            self._budget(job,rows,expected or 0)
            deadline = min([job['deadline'],*[a['expires_at'] for a in inputs]])
            body = UploadInput(project_id=job['project_id'],name=name,expected_bytes=expected,
                idempotency_key=f"job-{identity[0]}-{identity[2]}-{self.output_sequence(conn,identity)}")
            def register(connection, asset_id):
                # Recheck with the same transaction as registration/reservation.
                self._fence(connection,identity)
                connection.execute("INSERT INTO ptb_storage.job_assets(job_id,asset_id,owner_id,project_id,role,generation,created_at) VALUES(%s,%s,%s,%s,'output',%s,%s)",
                    (job['id'],asset_id,job['owner_id'],job['project_id'],identity[2],self.storage._now(connection)))
            return self.storage._create(conn,str(job['owner_id']),body,kind=kind,deadline=deadline,on_created=register)

    def output_limit(self,job):return 16

    def output_sequence(self,conn,identity):
        return len(FilePipeline._outputs(self,conn,identity))

    def independent_expiry(self,operation):return operation=='storage_check'

    def manifest(self,operation,files):return FileManifest(files=files,core_version=core_version).model_dump()

    @staticmethod
    def _budget(job, outputs, extra):
        budget = json.loads(job['snapshot'])['config']['max_output_bytes']
        if sum(a['size_bytes']+a['reserved_bytes'] for a in outputs)+extra > budget:
            raise StorageError('output_budget_exceeded',413)

    def write(self, identity, asset_id, offset, data):
        with self.storage._locked() as conn:
            job, _ = self._fence(conn,identity)
            output = self._output(conn,identity,asset_id)
            if output['sha256'] is not None: raise StorageError('output_sealed',409)
            self._budget(job,self._outputs(conn,identity),max(0,len(data)-output['reserved_bytes']))
            return self.storage._append(conn,str(job['owner_id']),asset_id,offset,data)

    def read_input(self, identity, asset_id, offset, size):
        if not 0 < size <= CHUNK_BYTES or offset < 0:
            raise StorageError('invalid_chunk',422)
        with self.storage._locked() as conn:
            job, inputs = self._fence(conn,identity)
            if str(asset_id) not in {str(a['id']) for a in inputs}:
                raise StorageError('asset_not_found',404)
            row = self.storage._readable(conn,str(job['owner_id']),asset_id)
            with self.storage._path(asset_id).open('rb') as handle:
                handle.seek(offset)
                data = handle.read(min(size,max(0,row['size_bytes']-offset)))
            self._fence(conn,identity)
            return data

    def seal(self, identity, asset_id):
        digest, offset = hashlib.sha256(), 0
        while True:
            with self.storage._locked() as conn:
                self._fence(conn,identity)
                row = self.storage._sync(conn,self._output(conn,identity,asset_id))
                if row['state']!='uploading': raise StorageError('output_closed',409)
                if row['expected_bytes'] is not None and row['size_bytes']!=row['expected_bytes']:
                    raise StorageError('upload_incomplete',409)
                with self.storage._path(asset_id).open('rb') as handle:
                    handle.seek(offset)
                    block = handle.read(CHUNK_BYTES)
                if not block:
                    conn.execute('UPDATE ptb_storage.assets SET sha256=%s WHERE id=%s',(digest.hexdigest(),asset_id))
                    return
                digest.update(block)
                offset += len(block)

    def complete(self, identity):
        with self.storage._locked() as conn, self.tx(conn) as tx:
            job, inputs = self._fence(conn,identity)
            outputs = self._outputs(conn,identity)
            if not outputs or any(a['state']!='uploading' or not a['sha256'] or a['kind']=='temporary' for a in outputs):
                raise StorageError('incomplete_output_batch',409)
            now = tx.now()
            operation = json.loads(job['snapshot'])['operation']
            deadline = expiry(now, [min(a['expires_at'],a['input_expires_at']) for a in inputs] if not self.independent_expiry(operation) else [])
            files = []
            for item in outputs:
                if self.storage._path(item['id']).stat().st_size != item['size_bytes']:
                    raise StorageError('storage_inconsistent',503)
                files.append(dict(id=str(item['id']),name=item['name'],kind=item['kind'],size_bytes=item['size_bytes'],sha256=item['sha256'],expires_at=deadline))
                conn.execute('UPDATE ptb_storage.quota_accounts SET reserved_bytes=reserved_bytes-%s WHERE owner_id=%s',(item['reserved_bytes'],job['owner_id']))
                conn.execute("UPDATE ptb_storage.assets SET state='ready',reserved_bytes=0,expires_at=%s WHERE id=%s",(deadline,item['id']))
            # Time can advance during the file checks/SQL above. Revalidate
            # immediately before publishing; an exception rolls back the whole
            # batch, including all ready flags and released reservations.
            self._fence(conn,identity)
            now=tx.now()
            deadline=expiry(now,[min(a['expires_at'],a['input_expires_at']) for a in inputs] if not self.independent_expiry(operation) else [])
            conn.execute('UPDATE ptb_storage.assets SET expires_at=%s WHERE id=ANY(%s)',(deadline,[a['id'] for a in outputs]))
            for item in files: item['expires_at']=deadline
            manifest = self.manifest(operation,files)
            tx.execute("UPDATE {jobs} SET state='succeeded',progress=1,result_manifest=?,error_code=NULL,worker_id=NULL,lease_until=NULL WHERE id=?",(canonical(manifest),job['id']))
            job.update(state='succeeded',progress=1)
            self.jobs._event(tx,job,'succeeded',now)
            return manifest

    def fail(self, identity, code):
        safe = {'cancelled','input_unavailable','quota_exceeded','output_budget_exceeded','archive_rejected',
                'disk_space_low','storage_write_failed','output_count_exceeded','stale_worker'}
        if code not in safe: code='execution_failed'
        with self.storage._locked() as conn:
            with self.tx(conn) as tx:
                self.jobs._recover(tx,tx.now())
                job = self.jobs._row(tx,identity[0])
                if job and job['generation']==identity[2] and job['worker_id']==identity[1] and job['state'] in ('running','cancel_requested'):
                    now = tx.now()
                    state = 'cancelled' if job['state']=='cancel_requested' or code=='cancelled' else 'failed'
                    tx.execute('UPDATE {jobs} SET state=?,error_code=?,worker_id=NULL,lease_until=NULL WHERE id=?',(state,code,job['id']))
                    job['state']=state
                    self.jobs._event(tx,job,code,now)
            for output in self._outputs(conn,identity):
                if output['state']!='ready': self.storage._delete(conn,output)

    def before_delete(self, conn, asset):
        with self.tx(conn) as tx:
            rows = conn.execute("SELECT j.* FROM ptb_jobs.jobs j JOIN ptb_storage.job_assets l ON l.job_id=j.id WHERE l.asset_id=%s AND j.state IN ('queued','running')",(asset['id'],)).fetchall()
            for value in rows:
                row = dict(value)
                row['state']=cancel_state(row['state'])
                tx.execute('UPDATE {jobs} SET state=?,error_code=? WHERE id=?',(row['state'],'resource_removed',row['id']))
                self.jobs._event(tx,row,'resource_removed',tx.now())

    def impact(self, owner, asset_id):
        with self.storage._locked() as conn:
            self.storage._row(conn,owner,asset_id)
            ids = [r['job_id'] for r in conn.execute("SELECT l.job_id FROM ptb_storage.job_assets l JOIN ptb_jobs.jobs j ON j.id=l.job_id WHERE l.asset_id=%s AND j.state IN ('queued','running','cancel_requested')",(asset_id,)).fetchall()]
            return {'active_jobs':ids}

    def reconcile(self, conn):
        with self.tx(conn) as tx:
            self.jobs._recover(tx,tx.now())
        rows = conn.execute("SELECT a.* FROM ptb_storage.assets a JOIN ptb_storage.job_assets l ON l.asset_id=a.id JOIN ptb_jobs.jobs j ON j.id=l.job_id WHERE l.role='output' AND a.state NOT IN ('ready','deleted') AND (j.state NOT IN ('queued','running') OR l.generation!=j.generation)").fetchall()
        for row in rows:
            self.storage._delete(conn,row)
