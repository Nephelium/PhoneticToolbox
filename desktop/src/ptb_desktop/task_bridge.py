"""Native capability to local API translation and non-overwriting result export."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
from uuid import UUID,uuid4
from .file_provider import checked_path,FileAccessError,identity

PROJECT='00000000-0000-4000-8000-000000000001'


@contextmanager
def pin_directory(path):
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW;create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];create.restype=wintypes.HANDLE
    handle=create(str(path),0x80000000,3,None,3,0x02200000,None)
    if handle==ctypes.c_void_p(-1).value:raise FileAccessError('无法锁定结果目录，请重新选择。')
    close=kernel.CloseHandle;close.argtypes=[wintypes.HANDLE]
    try:yield
    finally:close(handle)


class TaskBridge:
    def __init__(self,provider,service):self.provider,self.service=provider,service

    def invoke(self,body):
        op=body.get('op')
        if op=='list':return self.service.get('/api/v1/jobs/batches/list?project_id='+PROJECT)['batches']
        if op=='parent':
            _,sha=self.provider.read(body['id'])
            return self.service.get('/api/v1/jobs/parents/latest?project_id='+PROJECT+'&sha256='+sha)
        if op in ('get','cancel','job','retry'):
            key=str(UUID(body['id']))
            path='/api/v1/jobs/'+(key if op in ('job','retry') else 'batches/'+key)
            if op in ('cancel','retry'):path+='/'+op
            return self.service.request(path,'POST' if op in ('cancel','retry') else 'GET',{'idempotency_key':body['key']} if op=='retry' else None)
        if op=='submit':
            if set(body)-{'op','operation','inputs','config','layer','idempotency_key'} or not isinstance(body.get('inputs'),list) or not 1<=len(body['inputs'])<=1000:
                raise FileAccessError('批次请求不正确。')
            inputs=[]
            for item in body['inputs']:
                if set(item)-{'audio','textgrid','lip','parent_result'}:raise FileAccessError('不支持的关联。')
                mapped={}
                for role,key in item.items():
                    if key is None:continue
                    if role=='parent_result':
                        if set(key)!={'asset_id','sha256'}:raise FileAccessError('参数来源不正确。')
                        mapped[role]=key;continue
                    raw,_=self.provider.read(key);entry=self.provider.entries[key]
                    mapped[role]=self.service.import_input(raw,entry.name,role)
                inputs.append(mapped)
            return self.service.request('/api/v1/jobs/batches/create','POST',dict(project_id=PROJECT,operation=body['operation'],inputs=inputs,
                config=body.get('config'),layer=body.get('layer'),idempotency_key=body['idempotency_key']))
        if op=='save':return self.save(str(UUID(body['id'])),body['directory'])
        raise FileAccessError('不支持的任务操作。')

    def save(self,batch_id,directory_id):
        directory=self.provider.directory(directory_id)
        if directory.purpose not in ('input','output'):raise FileAccessError('请选择结果目录。')
        root=directory.path;batch=self.service.get('/api/v1/jobs/batches/'+batch_id)
        created=[];pending={};saved=[]
        def sha(path):
            value=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(65536),b''):value.update(block)
            return value.hexdigest()
        with pin_directory(root):
            self.provider.directory(directory_id)
            try:
                for item in batch['summary']['items']:
                    if item['state']!='succeeded':continue
                    job=self.service.get('/api/v1/jobs/'+item['job_id'])
                    for file in job['result_manifest']['files']:
                        name=file['name']
                        if job['operation']=='acoustic_analysis':name=Path(batch['audio_names'][item['index']]).stem+name[len('result'):]
                        if not name or len(name)>220 or any(ord(c)<32 or c in '/\\:<>"|?*' for c in name):raise FileAccessError('输出文件名不受支持。')
                        path=root/name
                        if path.exists():
                            checked_path(path)
                            if path.is_file() and path.stat().st_size==file['size_bytes'] and sha(path)==file['sha256']:
                                saved.append(name);continue
                            path=root/(item['job_id'][:8]+'-'+name)
                        if path.exists():
                            checked_path(path)
                            if path.is_file() and path.stat().st_size==file['size_bytes'] and sha(path)==file['sha256']:
                                saved.append(path.name);continue
                            path=root/(uuid4().hex[:8]+'-'+name)
                        temp=root/('.ptb-'+uuid4().hex+'.part')
                        digest=hashlib.sha256();offset=0
                        with temp.open('xb') as f:
                            pending[temp]=identity(os.fstat(f.fileno()))
                            while offset<file['size_bytes']:
                                self.provider.directory(directory_id)
                                size=min(1_048_576,file['size_bytes']-offset)
                                raw=self.service.binary(f'/api/v1/jobs/local-results/{file["id"]}?offset={offset}&size={size}')
                                if len(raw)!=size:raise FileAccessError('结果下载不完整。')
                                f.write(raw);digest.update(raw);offset+=size
                            f.flush();os.fsync(f.fileno())
                        if digest.hexdigest()!=file['sha256']:raise FileAccessError('结果校验失败。')
                        self.provider.directory(directory_id)
                        os.rename(temp,path)  # Windows refuses any pre-existing destination.
                        created.append((path,pending.pop(temp),file['sha256']));saved.append(path.name)
                return {'saved':saved,'count':len(saved),'batch_id':batch_id}
            except BaseException:
                for path,original in pending.items():
                    if path.exists() and identity(path.stat())==original:checked_path(path);path.unlink()
                for path,original,digest in created:
                    if path.exists() and identity(path.stat())==original:
                        checked_path(path)
                        if sha(path)==digest:path.unlink()
                raise
