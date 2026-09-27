"""Shared transactional job rules. Constructors and startup never run DDL."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from phonetic_core import __version__ as core_version
from ptb_api import __version__ as adapter_version
from ptb_api.job_models import JobInput, JobManifest, FileJobInput
from .policy import ACTIVE, cancel_state, fenced

LOCAL_PROJECT = '00000000-0000-4000-8000-000000000001'


class JobError(Exception):
    def __init__(self, code, status=409):
        self.code, self.status = code, status
        super().__init__(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False)


def public(row):
    snapshot=json.loads(row['snapshot'])
    return {k:row[k] for k in ('id','state','progress','generation','created_at','updated_at','error_code')} | {
        'project_id':str(row['project_id']), 'operation':snapshot['operation'],
        'retry_of':snapshot.get('retry_of'), 'waiting_reason':snapshot.get('waiting_reason'),
        'result_manifest':json.loads(row['result_manifest']) if row['result_manifest'] else None}


class Transaction:
    def __init__(self, conn, postgres):
        self.conn,self.postgres=conn,postgres

    def execute(self, query, params=()):
        query=query.replace('{jobs}','ptb_jobs.jobs' if self.postgres else 'jobs').replace(
            '{events}','ptb_jobs.events' if self.postgres else 'events')
        if self.postgres:query=query.replace('?', '%s')
        return self.conn.execute(query,params)

    def now(self):
        if self.postgres:
            return float(self.conn.execute('SELECT extract(epoch FROM clock_timestamp()) AS now').fetchone()['now'])
        return time.time()


class JobStore:
    """Mutating calls hold a short cross-process scheduling transaction, never computation."""
    postgres=False

    def __init__(self, *, max_running=2, lease_seconds=10):
        if not 1 <= max_running <= 16 or not 2 <= lease_seconds <= 60:
            raise ValueError('Invalid scheduler settings')
        self.max_running,self.lease_seconds=max_running,lease_seconds
        self.files = None
        self.batches = None

    def capacity_used(self,tx,owner):
        used=tx.execute('SELECT count(*) AS n FROM {jobs} WHERE owner_id=?',(owner,)).fetchone()['n']
        exists=(tx.execute("SELECT to_regclass('ptb_jobs.acoustic_batches') AS n").fetchone()['n'] if self.postgres else
                tx.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='acoustic_batches'").fetchone())
        if exists:
            prefix='ptb_jobs.' if self.postgres else ''
            used+=tx.execute(f'''SELECT count(*) AS n FROM {prefix}acoustic_batch_items i JOIN {prefix}acoustic_batches b
                ON b.id=i.batch_id WHERE b.owner_id=? AND b.closed_at IS NULL AND NOT b.cancel_requested AND i.child_job_id IS NULL''',(owner,)).fetchone()['n']
        return used

    def check_schema(self):
        with self.transaction(write=False) as tx:
            table='ptb_jobs.schema_version' if self.postgres else 'schema_version'
            if [dict(r) for r in tx.conn.execute(f'SELECT version FROM {table}').fetchall()] != [{'version':1}]:
                raise RuntimeError('P06 schema version mismatch')

    def _event(self, tx, row, code, now):
        sequence=row['event_seq']+1
        tx.execute('UPDATE {jobs} SET event_seq=?,updated_at=? WHERE id=?',(sequence,now,row['id']))
        tx.execute('INSERT INTO {events}(job_id,sequence,state,progress,code,created_at) VALUES (?,?,?,?,?,?)',
                   (row['id'],sequence,row['state'],row['progress'],code,now))

    def _row(self, tx, job_id, owner=None):
        query='SELECT * FROM {jobs} WHERE id=?'
        params=(job_id,)
        if owner is not None:query+=' AND owner_id=?';params+=(owner,)
        result=tx.execute(query,params).fetchone()
        return dict(result) if result else None

    def _recover(self, tx, now):
        rows=tx.execute("SELECT * FROM {jobs} WHERE (state IN ('running','cancel_requested') AND (lease_until<=? OR deadline<=?)) OR (state='queued' AND deadline<=?)",(now,now,now)).fetchall()
        for value in rows:
            row=dict(value)
            state='failed' if row['state']=='queued' else 'interrupted'
            code='deadline_exceeded' if row['deadline']<=now else 'worker_interrupted'
            tx.execute('UPDATE {jobs} SET state=?,error_code=?,worker_id=NULL,lease_until=NULL WHERE id=?',(state,code,row['id']))
            row['state']=state
            self._event(tx,row,code,now)

    def recover(self):
        with self.transaction() as tx:self._recover(tx,tx.now())

    def submit(self, owner, body: JobInput, *, retry_of=None):
        if isinstance(body, FileJobInput):
            if not self.files:
                raise JobError('file_tasks_unavailable',503)
            return self.files.submit(owner,body,retry_of=retry_of)
        request=body.model_dump(exclude={'idempotency_key'}) | {'retry_of':retry_of}
        request_hash=hashlib.sha256(canonical(request).encode('utf-8')).hexdigest()
        with self.transaction() as tx:
            old=tx.execute('SELECT * FROM {jobs} WHERE owner_id=? AND idempotency_key=?',(owner,body.idempotency_key)).fetchone()
            if old:
                if old['request_hash'] != request_hash:raise JobError('idempotency_conflict')
                return public(dict(old))
            if self.capacity_used(tx,owner) >= 1000:
                raise JobError('job_limit_reached')
            if self.postgres and not tx.conn.execute('SELECT 1 FROM ptb_accounts.projects p JOIN ptb_accounts.users u ON p.owner_id=u.id WHERE p.id=%s AND p.owner_id=%s AND u.active',(body.project_id,owner)).fetchone():
                raise JobError('project_not_found',404)
            if not self.postgres and (owner!='local' or body.project_id != LOCAL_PROJECT):
                raise JobError('project_not_found',404)
            snapshot=request | {'core_version':core_version,'adapter_version':adapter_version,'source_ids':[],
                                'input_sha256':hashlib.sha256(canonical(body.config.model_dump()).encode()).hexdigest()}
            now=tx.now(); job_id=str(uuid4())
            tx.execute("INSERT INTO {jobs}(id,owner_id,project_id,idempotency_key,request_hash,snapshot,state,deadline,created_at,updated_at) VALUES (?,?,?,?,?,?,'queued',?,?,?)",
                       (job_id,owner,body.project_id,body.idempotency_key,request_hash,canonical(snapshot),now+300,now,now))
            row=self._row(tx,job_id);self._event(tx,row,'queued',now)
            return public(row)

    def list(self, owner, project_id):
        with self.transaction() as tx:
            self._recover(tx,tx.now())
            return [public(dict(r)) for r in tx.execute('SELECT * FROM {jobs} WHERE owner_id=? AND project_id=? ORDER BY created_at DESC,id DESC LIMIT 100',(owner,project_id)).fetchall()]

    def get(self, owner, job_id):
        with self.transaction() as tx:
            self._recover(tx,tx.now());row=self._row(tx,job_id,owner)
            if not row:raise JobError('job_not_found',404)
            return public(row)

    def events(self, owner, job_id, after):
        with self.transaction() as tx:
            self._recover(tx,tx.now())
            if not self._row(tx,job_id,owner):raise JobError('job_not_found',404)
            return [dict(r) for r in tx.execute('SELECT sequence,state,progress,code,created_at FROM {events} WHERE job_id=? AND sequence>? ORDER BY sequence LIMIT 100',(job_id,after)).fetchall()]

    def cancel(self, owner, job_id):
        with self.transaction() as tx:
            now=tx.now();self._recover(tx,now);row=self._row(tx,job_id,owner)
            if not row:raise JobError('job_not_found',404)
            state=cancel_state(row['state'])
            if state != row['state']:
                tx.execute('UPDATE {jobs} SET state=? WHERE id=?',(state,job_id));row['state']=state
                self._event(tx,row,state,now)
            return public(self._row(tx,job_id))

    def retry(self, owner, job_id, key):
        with self.transaction() as tx:
            self._recover(tx,tx.now());row=self._row(tx,job_id,owner)
            if not row:raise JobError('job_not_found',404)
            if row['state'] not in ('failed','interrupted','cancelled'):raise JobError('job_not_retryable')
            snapshot=json.loads(row['snapshot'])
        if snapshot['operation']=='lip_analysis':
            from .m05_task import submit
            return submit(self,owner,dict({k:v for k,v in snapshot['request'].items() if k!='retry_of'},idempotency_key=key),retry_of=job_id)
        if snapshot['operation']=='mfa_alignment':
            from .m11_task import submit
            return submit(self,owner,dict({k:v for k,v in snapshot['request'].items() if k!='retry_of'},idempotency_key=key),retry_of=job_id)
        model = JobInput if snapshot['operation']=='pipeline_check' else FileJobInput
        if snapshot['operation']=='phonation_synthesis':
            from .m07_task import submit
            return submit(self,owner,dict({k:v for k,v in snapshot['request'].items() if k!='retry_of'},idempotency_key=key),retry_of=job_id)
        if snapshot['operation']=='speech_synthesis':
            from .m06_task import submit
            return submit(self,owner,dict(project_id=str(row['project_id']),idempotency_key=key,
                action=snapshot['config']['action'],audio=snapshot['input_refs'].get('audio'),
                parameters=snapshot['input_refs']['table']),retry_of=job_id)
        if snapshot['operation']=='pitch_manipulation':
            if snapshot.get('saved_copy'):
                raise JobError('m08_saved_copy_retry_requires_save',409)
            from .m08_task import submit
            return submit(self,owner,dict(project_id=str(row['project_id']),idempotency_key=key,
                audio=snapshot['input_refs']['audio'],config=snapshot['config']['analysis']),retry_of=job_id)
        if snapshot['operation']=='lpc_analysis':
            from .lpc_jobs import submit
            return submit(self,owner,dict(project_id=str(row['project_id']),idempotency_key=key,
                audio=snapshot['input_refs']['audio'],textgrid=snapshot['input_refs'].get('textgrid'),
                config=snapshot['config']['analysis']),retry_of=job_id)
        if snapshot['operation']=='egg_analysis':
            from .egg_jobs import submit
            return submit(self,owner,dict(project_id=str(row['project_id']),idempotency_key=key,
                audio=snapshot['input_refs']['audio'],config=snapshot['config']['analysis']),retry_of=job_id)
        if snapshot['operation']=='spectrogram_to_audio':
            from .spec2wav_jobs import submit
            return submit(self,owner,dict(project_id=str(row['project_id']),idempotency_key=key,image=snapshot['input_refs']['image'],config=snapshot['config']['analysis']),retry_of=job_id)
        if snapshot['operation'] in ('acoustic_analysis','textgrid_segment'):
            if self.batches is None:raise JobError('acoustic_tasks_unavailable',503)
            return self.batches.retry(owner,job_id,key)
        if model is FileJobInput:
            from ptb_api.storage_policy import QUOTA_BYTES
            # A retry is a new request. Keep the old snapshot intact and return
            # a public error instead of leaking an internal validation failure.
            if snapshot['config'].get('max_output_bytes', 0) > QUOTA_BYTES:
                raise JobError('output_budget_exceeded',413)
        body=model(project_id=str(row['project_id']),idempotency_key=key,operation=snapshot['operation'],config=snapshot['config'])
        return self.submit(owner,body,retry_of=job_id)

    def claim(self, worker_id):
        if self.files is not None:
            return self.files.claim(worker_id)
        with self.transaction() as tx:
            return self._claim(tx,worker_id)

    def _claim(self, tx, worker_id):
        now=tx.now();self._recover(tx,now)
        if self.batches is not None:self.batches.advance(tx,now)
        if tx.execute("SELECT count(*) AS n FROM {jobs} WHERE state IN ('running','cancel_requested')").fetchone()['n'] >= self.max_running:return None
        query="""SELECT j.* FROM {jobs} j WHERE j.state='queued' AND j.snapshot NOT LIKE ? AND NOT EXISTS
            (SELECT 1 FROM {jobs} a WHERE a.owner_id=j.owner_id AND a.state IN ('running','cancel_requested'))
            ORDER BY j.created_at,j.id LIMIT 1"""
        params=('%"execution_route":"remote_pending"%',)
        if self.files is None:
            query=query.replace('ORDER BY j.created_at','AND j.snapshot LIKE ? ORDER BY j.created_at')
            params=params+('%"operation":"pipeline_check"%',)
        if self.postgres:query+=' FOR UPDATE OF j SKIP LOCKED'
        found=tx.execute(query,params).fetchone()
        if not found:return None
        row=dict(found);generation=row['generation']+1
        tx.execute("UPDATE {jobs} SET state='running',worker_id=?,generation=?,lease_until=? WHERE id=?",(worker_id,generation,now+self.lease_seconds,row['id']))
        row=self._row(tx,row['id']);self._event(tx,row,'running',now)
        return row

    def heartbeat(self, job_id, worker_id, generation, progress):
        if not 0 <= progress < 1:raise ValueError('Only committed results reach progress 1')
        with self.transaction() as tx:
            now=tx.now();row=self._row(tx,job_id)
            if not row or not fenced(row,worker_id,generation,now):return None
            if row['state']=='cancel_requested':return 'cancel_requested'
            progress=max(row['progress'],min(0.99,round(progress,2)))
            tx.execute('UPDATE {jobs} SET lease_until=?,progress=? WHERE id=?',(now+self.lease_seconds,progress,job_id))
            if progress > row['progress']:
                row['progress']=progress;self._event(tx,row,'progress',now)
            return 'running'

    def finish(self, job_id, worker_id, generation, *, result=None, error=None):
        manifest=JobManifest.model_validate(result).model_dump() if result is not None else None
        if error not in (None,'execution_failed','cancelled','core_version_mismatch'):raise ValueError('Unsafe error code')
        with self.transaction() as tx:
            now=tx.now();row=self._row(tx,job_id)
            if not row or not fenced(row,worker_id,generation,now):return False
            if row['state']=='cancel_requested' or error=='cancelled':state='cancelled';manifest=None;error=None
            elif manifest is not None:
                snapshot=json.loads(row['snapshot'])
                if manifest['core_version'] != snapshot['core_version'] or manifest['sample_count'] != snapshot['config']['sample_count']:
                    raise ValueError('Manifest does not match immutable snapshot')
                state='succeeded'
            else:state='failed';error=error or 'execution_failed'
            progress=1.0 if state=='succeeded' else row['progress']
            tx.execute('UPDATE {jobs} SET state=?,progress=?,result_manifest=?,error_code=?,worker_id=NULL,lease_until=NULL WHERE id=?',
                       (state,progress,canonical(manifest) if manifest else None,error,job_id))
            row['state']=state;row['progress']=progress;self._event(tx,row,error or state,now)
            return True


class PostgresJobStore(JobStore):
    postgres=True

    def __init__(self, dsn, **kwargs):
        super().__init__(**kwargs);self._dsn=dsn

    @contextmanager
    def transaction(self, write=True):
        with psycopg.connect(self._dsn,row_factory=dict_row,connect_timeout=5,
                             options='-c statement_timeout=10000 -c lock_timeout=5000') as conn:
            if write:conn.execute('SELECT pg_advisory_xact_lock(577606)')
            yield Transaction(conn,True)


class SQLiteJobStore(JobStore):
    def __init__(self, path, **kwargs):
        super().__init__(**kwargs);self.path=Path(path).resolve()

    @contextmanager
    def transaction(self, write=True):
        # mode=rw prevents silent database creation. DDL is an explicit reviewed tool.
        conn=sqlite3.connect(self.path.as_uri()+'?mode=rw',uri=True,timeout=5)
        conn.row_factory=sqlite3.Row
        try:
            conn.execute('PRAGMA foreign_keys=ON')
            if write:conn.execute('BEGIN IMMEDIATE')
            yield Transaction(conn,False)
            conn.commit()
        except BaseException:
            conn.rollback();raise
        finally:conn.close()
