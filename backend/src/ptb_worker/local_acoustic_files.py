"""Offline managed assets; SQLite success is the visibility gate for output files.

Explicit initialization is separate from construction. Paths never enter renderer
requests; inputs are copies obtained through native directory capabilities.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from uuid import UUID,uuid4
from .io.scratch import no_links
from .store import JobError,LOCAL_PROJECT,canonical,core_version
from .policy import fenced,cancel_state
from ptb_api.quota import StorageError
from ptb_api.acoustic_batch_models import AcousticTaskManifest


def initialize_local_files(root):
    root=Path(root).absolute();no_links(root)
    if not root.is_dir() or any(root.iterdir()):raise ValueError('Initialization requires a new empty owned directory')
    (root/'.ptb-local.lock').write_bytes(b'0')
    (root/'.ptb-local.json').write_text(canonical(dict(version=1,instance_id=str(uuid4()),assets={})),encoding='utf-8')


class LocalAcousticFiles:
    def __init__(self,jobs,root,*,reaper_binary=None):
        if jobs.postgres:raise ValueError('Local SQLite required')
        self.jobs=jobs;self.root=Path(root).absolute();no_links(self.root)
        self.reaper_binary=reaper_binary;self.thread_lock=threading.RLock()
        with self.locked():pass
        jobs.files=self

    @property
    def scratch_root(self):return self.root

    def _path(self,asset_id):
        path=self.root/(str(UUID(asset_id))+'.bin');no_links(path)
        if path.exists() and (not path.is_file() or path.stat().st_nlink!=1):raise StorageError('local_path_rejected',403)
        return path

    def _save(self,state):
        path=self.root/'.ptb-local-next.json';no_links(path)
        raw=canonical(state).encode()
        if len(raw)>16_000_000:raise StorageError('local_metadata_limit',413)
        with path.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(path,self.root/'.ptb-local.json')

    @contextmanager
    def locked(self):
        import msvcrt
        with self.thread_lock:
            no_links(self.root);path=self.root/'.ptb-local.lock';no_links(path)
            with path.open('r+b',buffering=0) as lock:
                deadline=time.monotonic()+5
                while True:
                    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
                    except OSError:
                        if time.monotonic()>=deadline:raise StorageError('local_storage_busy',503) from None
                        time.sleep(.01)
                try:
                    marker=self.root/'.ptb-local.json';no_links(marker)
                    if marker.stat().st_size>16_000_000:raise StorageError('local_metadata_limit',413)
                    state=json.loads(marker.read_text('utf-8'))
                    if state['version']!=1:raise StorageError('local_schema_mismatch',503)
                    self.state=state
                    yield state
                finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

    @contextmanager
    def batch_transaction(self):
        with self.locked(),self.jobs.transaction() as tx:yield tx

    def _asset(self,asset_id):
        row=self.state['assets'].get(str(asset_id))
        if row is None:raise JobError('input_unavailable',404)
        return row

    def _budget(self,extra):
        reserved=sum(a['reserved_bytes'] for a in self.state['assets'].values())
        if len(self.state['assets'])>=10000:raise StorageError('asset_limit_reached',413)
        if shutil.disk_usage(self.root).free-reserved-extra<1_000_000_000:raise StorageError('disk_space_low',507)

    def import_input(self,raw,name,role):
        suffix={'audio':'.wav','textgrid':'.textgrid','lip':'.lip.json','parent_result':'.ptb.json','legacy_result':('.xlsx','.ptb.sqlite','.ptb.sqlite3')}.get(role)
        limit=64_000_000 if role=='audio' else 16_000_000 if role in ('parent_result','legacy_result') else 2_000_000
        if not suffix or not 0<len(raw)<=limit or not isinstance(name,str) or not 0<len(name)<=220 or any(c in name for c in '/\\:\x00') or not name.lower().endswith(suffix):
            raise JobError('invalid_local_input',422)
        sha=hashlib.sha256(raw).hexdigest()
        with self.locked() as state:
            for a in state['assets'].values():
                if a['state']=='ready' and a['kind']=='input' and (a['sha256'],a['name'],a['role'])==(sha,name,role):
                    return dict(asset_id=a['id'],sha256=sha)
            self._budget(len(raw));asset_id=str(uuid4())
            row=dict(id=asset_id,name=name,role=role,kind='input',state='uploading',sha256=sha,size_bytes=0,
                     expected_bytes=len(raw),reserved_bytes=len(raw),job_id=None,generation=0)
            state['assets'][asset_id]=row;self._save(state)
            with self._path(asset_id).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            row.update(state='ready',size_bytes=len(raw),reserved_bytes=0);self._save(state)
            return dict(asset_id=asset_id,sha256=sha)

    def validate_batch_input(self,tx,owner,project,inputs,now):
        if owner!='local' or project!=LOCAL_PROJECT:raise JobError('project_not_found',404)
        result=[]
        for role,ref in inputs.items():
            if ref is None:continue
            a=self._asset(ref['asset_id'])
            if a['state']!='ready' or a['sha256']!=ref['sha256'] or (role!='parent_result' and a['role']!=role):
                raise JobError('input_unavailable',409)
            if role=='parent_result' and not a['name'].endswith('.ptb.json'):raise JobError('invalid_parent_result',422)
            if role=='parent_result':
                producer=self.jobs._row(tx,a['job_id']) if a['job_id'] else None
                if not producer or json.loads(producer['snapshot'])['operation']!='acoustic_analysis':raise JobError('invalid_parent_result',422)
            try:self._readable(tx,a)
            except (StorageError,OSError):raise JobError('input_unavailable',409) from None
            result.append(dict(id=a['id'],role=role,sha256=a['sha256'],expires_at=None,size_bytes=a['size_bytes'],name=a['name']))
        return result

    def link_batch_inputs(self,*args):pass  # Immutable P06 snapshot records the local references.

    def _readable(self,tx,row):
        if row['state']!='ready':raise StorageError('input_unavailable',410)
        if row['job_id']:
            job=self.jobs._row(tx,row['job_id'])
            if not job or job['state']!='succeeded' or not any(f['id']==row['id'] for f in json.loads(job['result_manifest'])['files']):
                raise StorageError('output_not_committed',409)
        if self._path(row['id']).stat().st_size!=row['size_bytes']:raise StorageError('input_unavailable',410)

    def _fence(self,tx,identity):
        now=tx.now();job=self.jobs._row(tx,identity[0])
        if not job or not fenced(job,identity[1],identity[2],now):raise StorageError('stale_worker',409)
        if job['state']!='running':raise StorageError('cancelled',409)
        snapshot=json.loads(job['snapshot'])
        for item in snapshot['input_assets']:
            a=self._asset(item['id']);self._readable(tx,a)
            if a['sha256']!=item['sha256']:raise StorageError('input_unavailable',410)
        tx.execute('UPDATE {jobs} SET lease_until=? WHERE id=?',(now+self.jobs.lease_seconds,job['id']))
        return job

    def claim(self,worker_id):
        with self.batch_transaction() as tx:return self.jobs._claim(tx,worker_id)

    def heartbeat(self,identity):
        with self.batch_transaction() as tx:self._fence(tx,identity)

    def _outputs(self,identity):
        return [a for a in self.state['assets'].values() if a['job_id']==identity[0] and a['generation']==identity[2] and a['state']!='deleted']

    def _output(self,identity,asset_id):
        a=self._asset(asset_id)
        if a not in self._outputs(identity):raise StorageError('worker_owned_asset',403)
        return a

    def output(self,identity,name,kind,expected):
        if kind not in ('temporary','result') or type(expected)!=int or expected<=0:raise StorageError('invalid_output',422)
        with self.batch_transaction() as tx:
            job=self._fence(tx,identity);outputs=self._outputs(identity);self._budget(expected)
            if len(outputs)>=3004 or sum(a['size_bytes']+a['reserved_bytes'] for a in outputs)+expected>json.loads(job['snapshot'])['config']['max_output_bytes']:
                raise StorageError('output_budget_exceeded',413)
            asset_id=str(uuid4());a=dict(id=asset_id,name=name,role=None,kind=kind,state='uploading',sha256=None,size_bytes=0,
                reserved_bytes=expected,expected_bytes=expected,job_id=job['id'],generation=identity[2])
            self.state['assets'][asset_id]=a;self._save(self.state)
            with self._path(asset_id).open('xb') as f:f.flush();os.fsync(f.fileno())
            return dict(a)

    def write(self,identity,asset_id,offset,raw):
        with self.batch_transaction() as tx:
            self._fence(tx,identity);a=self._output(identity,asset_id)
            if a['state']!='uploading' or a['sha256'] is not None or offset!=a['size_bytes'] or not 0<len(raw)<=1_048_576 or offset+len(raw)>a['expected_bytes']:
                raise StorageError('invalid_output_chunk',409)
            with self._path(asset_id).open('ab') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            a['size_bytes']+=len(raw);a['reserved_bytes']-=len(raw);self._save(self.state)

    def read_input(self,identity,asset_id,offset,size):
        with self.batch_transaction() as tx:
            job=self._fence(tx,identity)
            if asset_id not in json.loads(job['snapshot'])['config']['inputs'] or not 0<size<=1_048_576 or offset<0:
                raise StorageError('input_unavailable',403)
            with self._path(asset_id).open('rb') as f:f.seek(offset);return f.read(size)

    def seal(self,identity,asset_id):
        with self.batch_transaction() as tx:
            self._fence(tx,identity);a=self._output(identity,asset_id)
            if a['size_bytes']!=a['expected_bytes'] or self._path(asset_id).stat().st_size!=a['size_bytes']:raise StorageError('upload_incomplete',409)
            digest=hashlib.sha256()
            with self._path(asset_id).open('rb') as f:
                for block in iter(lambda:f.read(65536),b''):digest.update(block)
            a['sha256']=digest.hexdigest();self._save(self.state)

    def scratch_path(self,identity,asset_id):
        with self.batch_transaction() as tx:
            self._fence(tx,identity);a=self._output(identity,asset_id)
            if a['kind']!='temporary':raise StorageError('not_scratch',403)
            return self._path(asset_id)

    def _delete(self,a):
        a['state']='deleting';self._save(self.state)
        self._path(a['id']).unlink(missing_ok=True)
        a.update(state='deleted',size_bytes=0,reserved_bytes=0);self._save(self.state)

    def release_scratch(self,identity,asset_id):
        with self.locked():
            a=self._output(identity,asset_id)
            if a['kind']!='temporary':raise StorageError('not_scratch',403)
            self._delete(a)

    def complete(self,identity):
        with self.batch_transaction() as tx:
            job=self._fence(tx,identity);outputs=self._outputs(identity)
            if not outputs or any(a['kind']!='result' or a['state']!='uploading' or not a['sha256'] for a in outputs):
                raise StorageError('incomplete_output_batch',409)
            for a in outputs:
                if self._path(a['id']).stat().st_size!=a['size_bytes']:raise StorageError('output_changed',409)
            manifest=AcousticTaskManifest(operation=json.loads(job['snapshot'])['operation'],core_version=core_version,
                files=[dict(id=a['id'],name=a['name'],kind='result',size_bytes=a['size_bytes'],sha256=a['sha256'],expires_at=None) for a in outputs]).model_dump()
            self._fence(tx,identity)
            for a in outputs:a.update(state='ready',reserved_bytes=0)
            self._save(self.state)
            tx.execute("UPDATE {jobs} SET state='succeeded',progress=1,result_manifest=?,error_code=NULL,worker_id=NULL,lease_until=NULL WHERE id=?",(canonical(manifest),job['id']))
            job.update(state='succeeded',progress=1);self.jobs._event(tx,job,'succeeded',tx.now())
            return manifest

    def fail(self,identity,code):
        with self.batch_transaction() as tx:
            self.jobs._recover(tx,tx.now());job=self.jobs._row(tx,identity[0])
            if job and job['generation']==identity[2] and job['worker_id']==identity[1] and job['state'] in ('running','cancel_requested'):
                state='cancelled' if job['state']=='cancel_requested' or code=='cancelled' else 'failed'
                from .acoustic_errors import ACOUSTIC_ERRORS
                code=code if code in ACOUSTIC_ERRORS | {'cancelled','input_unavailable','output_budget_exceeded','disk_space_low'} else 'execution_failed'
                tx.execute('UPDATE {jobs} SET state=?,error_code=?,worker_id=NULL,lease_until=NULL WHERE id=?',(state,code,job['id']))
                job['state']=state;self.jobs._event(tx,job,code,tx.now())
            for a in self._outputs(identity):
                if not job or job['state']!='succeeded' or job['generation']!=identity[2]:self._delete(a)

    def read_result(self,owner,asset_id,offset,size):
        if owner!='local' or offset<0 or not 0<size<=1_048_576:raise StorageError('invalid_read',403)
        with self.batch_transaction() as tx:
            a=self._asset(asset_id);self._readable(tx,a)
            with self._path(asset_id).open('rb') as f:f.seek(offset);return f.read(size)

    def recover(self):
        with self.batch_transaction() as tx:
            self.jobs._recover(tx,tx.now())
            for a in self.state['assets'].values():
                if a['state']=='deleted':continue
                job=self.jobs._row(tx,a['job_id']) if a['job_id'] else None
                if (a['job_id'] and (not job or job['generation']!=a['generation'] or job['state'] not in ('running','queued','succeeded'))) or (not a['job_id'] and a['state']!='ready'):
                    self._delete(a)
                elif job and job['state']=='succeeded':self._readable(tx,a)
