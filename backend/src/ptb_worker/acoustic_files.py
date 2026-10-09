"""M01 PostgreSQL adapter. P07 resources, quota, fencing and publication are reused."""
from contextlib import contextmanager
import json
from .files import FilePipeline
from .store import JobError,core_version
from .acoustic_batches import OPERATIONS
from .acoustic_errors import ACOUSTIC_ERRORS
from ptb_api.quota import StorageError
from ptb_api.storage_policy import POLICY_VERSION
from ptb_api.acoustic_batch_models import AcousticTaskManifest


class AcousticFiles(FilePipeline):
    additional_error_codes=ACOUSTIC_ERRORS

    @property
    def scratch_root(self):return self.storage.root

    @contextmanager
    def batch_transaction(self):
        with self.storage._locked() as conn,self.tx(conn) as tx:
            self.storage._writable(conn)
            yield tx

    def validate_batch_input(self,tx,owner,project,inputs,now):
        result=[]
        try:
            for role,ref in inputs.items():
                if ref is None:continue
                item=self.storage._readable(tx.conn,owner,ref['asset_id'])
                if str(item['project_id'])!=project or item['sha256']!=ref['sha256'] or item['expires_at']<=now+300:
                    raise JobError('input_unavailable',409)
                limit=32_000_000 if role=='m07_analysis' else 64_000_000 if role=='audio' else 16_000_000 if role=='dictionary' else 16_000_000 if role in ('parent_result','legacy_result','image') else 2_000_000
                suffix={'m07_analysis':'.m07.json','spectral_drawing':'.m09-drawing.json','transcript':('.lab','.txt','.textgrid'),'dictionary':('.dict','.txt'),'table':('.xlsx','.xls','.csv','.txt','.tsv','.docx'),'audio':('.wav','.mp3','.flac'),'textgrid':'.textgrid','lip':'.lip.json','parent_result':'.ptb.json','legacy_result':('.xlsx','.ptb.sqlite','.ptb.sqlite3'),'image':('.png','.jpg','.jpeg','.bmp')}.get(role)
                if not suffix:raise JobError('invalid_input',422)
                if item['size_bytes']>limit or not item['name'].lower().endswith(suffix):raise JobError('invalid_input',422)
                if role=='parent_result':
                    producer=tx.execute("SELECT j.snapshot,j.result_manifest FROM ptb_storage.job_assets l JOIN {jobs} j ON j.id=l.job_id WHERE l.asset_id=? AND l.role='output' AND j.state='succeeded'",(item['id'],)).fetchone()
                    if (not producer or json.loads(producer['snapshot'])['operation']!='acoustic_analysis' or
                        not any(f['id']==str(item['id']) for f in json.loads(producer['result_manifest'])['files'])):
                        raise JobError('invalid_parent_result',422)
                result.append(dict(id=str(item['id']),role=role,sha256=item['sha256'],expires_at=item['expires_at'],
                                   size_bytes=item['size_bytes'],name=item['name']))
        except StorageError:raise JobError('input_unavailable',409) from None
        return result

    def link_batch_inputs(self,tx,job_id,batch,inputs,now):
        for item in inputs:
            tx.execute('''INSERT INTO ptb_storage.job_assets(job_id,asset_id,owner_id,project_id,role,generation,
                input_sha256,input_expires_at,created_at) VALUES(?,?,?,?,'input',0,?,?,?)''',
                (job_id,item['id'],batch['owner_id'],batch['project_id'],item['sha256'],item['expires_at'],now))

    def _outputs(self,conn,identity):
        return [row for row in super()._outputs(conn,identity) if row['state']!='deleted']

    def output_limit(self,job):
        return 3004 if json.loads(job['snapshot'])['operation'] in (*OPERATIONS,'pitch_manipulation','mfa_alignment','phonation_synthesis') else super().output_limit(job)

    def manifest(self,operation,files):
        if operation=='mfa_alignment':
            from ptb_api.m11_models import M11Manifest
            return M11Manifest(policy_version=POLICY_VERSION,files=files,core_version=core_version).model_dump()
        if operation=='phonation_synthesis':
            from ptb_api.m07_models import M07Manifest
            return M07Manifest(policy_version=POLICY_VERSION,files=files,core_version=core_version).model_dump()
        if operation=='speech_synthesis':
            from ptb_api.m06_models import M06Manifest
            return M06Manifest(policy_version=POLICY_VERSION,files=files,core_version=core_version).model_dump()
        if operation=='phonology_induction':
            from ptb_api.m14_models import M14Manifest
            return M14Manifest(policy_version=POLICY_VERSION,files=files,core_version=core_version).model_dump()
        if operation=='pitch_manipulation':
            from ptb_api.m08_models import M08Manifest
            return M08Manifest(policy_version=POLICY_VERSION,files=files,core_version=core_version).model_dump()
        if operation=='lpc_analysis':
            from ptb_api.lpc_models import LpcManifest
            return LpcManifest(policy_version=POLICY_VERSION,files=files,core_version=core_version).model_dump()
        if operation=='egg_analysis':
            from ptb_api.egg_models import EggManifest
            return EggManifest(policy_version=POLICY_VERSION,files=files,core_version=core_version).model_dump()
        if operation=='spectrogram_to_audio':
            from ptb_api.spec2wav_models import Spec2WavManifest
            return Spec2WavManifest(policy_version=POLICY_VERSION,files=files,core_version=core_version).model_dump()
        if operation in OPERATIONS:return AcousticTaskManifest(policy_version=POLICY_VERSION,operation=operation,files=files,core_version=core_version).model_dump()
        return super().manifest(operation,files)

    def scratch_path(self,identity,asset_id):
        with self.storage._locked() as conn:
            self._fence(conn,identity);row=self._output(conn,identity,asset_id)
            if row['kind']!='temporary':raise StorageError('not_scratch',403)
            return self.storage._path(asset_id)

    def release_scratch(self,identity,asset_id):
        with self.storage._locked() as conn:
            row=self._output(conn,identity,asset_id)
            if row['kind']!='temporary':raise StorageError('not_scratch',403)
            result=self.storage._delete(conn,row,notify_jobs=False)
            if result['state']!='deleted':raise StorageError('scratch_cleanup_failed',503)

    def heartbeat(self,identity):
        with self.storage._locked() as conn:self._fence(conn,identity)
