"""M14 native boundary: imported bytes, durable task transport, atomic result set."""
import base64
import hashlib
import os
from uuid import UUID,uuid4
from .file_provider import FileAccessError,checked_path,identity
from .task_bridge import PROJECT,pin_directory
from phonetic_core.transcription.phonology.models import NAMES


class M14Bridge:
    def __init__(self,bridge):self.bridge=bridge

    def invoke(self,body):
        if body['op']=='m14_import':
            if not isinstance(body.get('base64'),str) or len(body['base64'])>2_666_668:raise FileAccessError('文件超过 2 MB。')
            raw=base64.b64decode(body['base64'],validate=True)
            return dict(self.bridge.service.import_input(raw,body['name'],'table'),name=body['name'])
        if body['op']=='m14_create':
            if body['body'].get('project_id')!=PROJECT:raise FileAccessError('项目不匹配。')
            return self.bridge.service.request('/api/v1/jobs/m14/create','POST',body['body'])
        if body['op']=='m14_save':return self.save(body['job'],body['directory'])
        raise FileAccessError('M14 操作不可用。')

    def save(self,job_id,directory_id):
        provider=self.bridge.provider;directory=provider.directory(directory_id)
        if directory.purpose!='output':raise FileAccessError('请选择输出目录。')
        job=self.bridge.service.get('/api/v1/jobs/'+str(UUID(job_id)))
        if job['state']!='succeeded' or job['operation']!='phonology_induction':raise FileAccessError('结果尚不可用。')
        files=job['result_manifest']['files']
        if sorted(f['name'] for f in files)!=sorted(NAMES):raise FileAccessError('三份结果不完整。')
        # Download/verify the entire bounded set before touching the destination.
        data=[]
        for f in files:
            value=self.bridge.invoke(dict(op='result',job=job_id,id=f['id']))
            raw=base64.b64decode(value['base64']);data.append((f['name'],raw))
        root=directory.path;pending={};created={}
        with pin_directory(root):
            provider.directory(directory_id)
            if any((root/n).exists() for n in NAMES):raise FileAccessError('输出目录已有同名结果，请选择另一个目录；原文件保留。')
            try:
                for name,raw in data:
                    temp=root/('.ptb-m14-'+uuid4().hex+'.part')
                    with temp.open('xb') as stream:
                        pending[temp]=identity(os.fstat(stream.fileno()));stream.write(raw);stream.flush();os.fsync(stream.fileno())
                    if hashlib.sha256(temp.read_bytes()).digest()!=hashlib.sha256(raw).digest():raise FileAccessError('保存校验失败。')
                for (temp,ident),(name,raw) in zip(list(pending.items()),data):
                    provider.directory(directory_id);target=root/name
                    os.rename(temp,target)  # Windows non-overwriting atomic rename.
                    created[target]=(ident,hashlib.sha256(raw).digest());del pending[temp]
                return dict(count=3,saved=list(NAMES))
            except BaseException:
                for p,ident in pending.items():
                    if p.exists() and identity(p.stat())==ident:checked_path(p);p.unlink()
                for p,(ident,digest) in created.items():
                    if p.exists() and identity(p.stat())==ident and hashlib.sha256(p.read_bytes()).digest()==digest:checked_path(p);p.unlink()
                raise
