"""Native capability to local API translation and non-overwriting result export."""
from contextlib import contextmanager
import hashlib
import json
import os
import struct
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
    def __init__(self,provider,service):
        self.provider,self.service=provider,service
        from .annotation import AnnotationFiles
        self.annotation=AnnotationFiles(provider)

    def invoke(self,body):
        op=body.get('op')
        if isinstance(op,str) and op.startswith('annotation_'):return self.annotation.invoke(body)
        if op=='lpc_fonts':return self.service.request('/api/v1/jobs/lpc/fonts','POST',body['font'])
        if op=='lpc':
            raw,_=self.provider.read(body['id']);entry=self.provider.entries[body['id']]
            ref=self.service.import_input(raw,entry.name,'audio');grid=None
            if body.get('textgrid'):
                data,_=self.provider.read(body['textgrid']);grid_entry=self.provider.entries[body['textgrid']]
                grid=self.service.import_input(data,grid_entry.name,'textgrid')
            return self.service.request('/api/v1/jobs/lpc/create','POST',dict(project_id=PROJECT,
                idempotency_key=body['key'],audio=ref,textgrid=grid,config=body['config']))
        if op=='lpc_jobs':return [j for j in self.service.get('/api/v1/jobs?project_id='+PROJECT)['jobs'] if j['operation']=='lpc_analysis']
        if op=='egg_fonts':return self.service.request('/api/v1/jobs/egg/fonts','POST',body['font'])
        if op=='egg':
            raw,_=self.provider.read(body['id']);entry=self.provider.entries[body['id']]
            ref=self.service.import_input(raw,entry.name,'audio')
            return self.service.request('/api/v1/jobs/egg/create','POST',dict(project_id=PROJECT,
                idempotency_key=body['key'],audio=ref,config=body['config']))
        if op=='egg_jobs':return [j for j in self.service.get('/api/v1/jobs?project_id='+PROJECT)['jobs'] if j['operation']=='egg_analysis']
        if op=='reconstruct':
            raw,_=self.provider.read(body['id']);entry=self.provider.entries[body['id']]
            ref=self.service.import_input(raw,entry.name,'image')
            return self.service.request('/api/v1/jobs/spec2wav/create','POST',dict(project_id=PROJECT,idempotency_key=body['key'],image=ref,config=body['config']))
        if op=='reconstructions':return [j for j in self.service.get('/api/v1/jobs?project_id='+PROJECT)['jobs'] if j['operation']=='spectrogram_to_audio']
        if op=='cancel_job':return self.service.request('/api/v1/jobs/'+str(UUID(body['id']))+'/cancel','POST')
        if op=='save_job':return self.save(str(UUID(body['id'])),body['directory'],single=True)
        if op=='result':
            import base64
            job=self.service.get('/api/v1/jobs/'+str(UUID(body['job'])))
            if job['state']!='succeeded' or job['operation'] not in ('spectrogram_to_audio','egg_analysis','lpc_analysis'):raise FileAccessError('分析结果尚不可用。')
            file=next((f for f in job['result_manifest']['files'] if f['id']==body['id']),None)
            if not file or file['size_bytes']>64_000_000:raise FileAccessError('结果文件不正确。')
            chunks=[self.service.binary(f'/api/v1/jobs/local-results/{file["id"]}?offset={offset}&size={min(1048576,file["size_bytes"]-offset)}') for offset in range(0,file['size_bytes'],1048576)]
            raw=b''.join(chunks)
            if len(raw)!=file['size_bytes'] or hashlib.sha256(raw).hexdigest()!=file['sha256']:raise FileAccessError('结果校验失败。')
            return {'base64':base64.b64encode(raw).decode(),'sha256':file['sha256']}
        if op=='parameters':
            raw,sha=self.provider.read(body['id']);entry=self.provider.entries[body['id']]
            from urllib.error import HTTPError
            try:result=self.service.parameters(raw,entry.name)
            except HTTPError as exc:
                code=json.load(exc).get('detail','')
                messages={'parameter_input_budget':'参数文件超过 16 MB。','parameter_read_timeout':'参数表读取超时，请使用较小的表。','preview_busy':'已有预览正在读取，请稍后重试。','invalid_parameter_table':'参数表格式不受支持、包含公式或超过显示预算。原文件未修改。','parameter_read_failed':'参数读取子进程未能完成。请使用完整研究工作台修复版，原文件未修改。','preview_platform_unverified':'此平台的参数读取尚未验证。'}
                raise FileAccessError(messages.get(code,'参数读取服务失败。请检查本机后台，不能据此判断原参数文件损坏。')) from None
            if result['sha256']!=sha:raise FileAccessError('参数文件校验失败。')
            return result
        if op=='convert_lip':return self.convert_lip(body['id'])
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
                if set(item)-{'audio','textgrid','lip','parent_result','legacy_result'}:raise FileAccessError('不支持的关联。')
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

    def convert_lip(self,file_id):
        raw,_=self.provider.read(file_id);entry=self.provider.entries[file_id]
        if not entry.name.lower().endswith('.pkl') or entry.name.lower().endswith('_timestamps.pkl'):raise FileAccessError('请选择旧唇形 PKL。')
        directory=self.provider.directory(entry.directory);root=directory.path
        name=entry.name[:-4]+'.lip.json';target=root/name
        if len(name)>220:raise FileAccessError('转换文件名过长。')
        companions=[f for f in self.provider.list(entry.directory) if f['name'].casefold()==(entry.name[:-4]+'_timestamps.pkl').casefold()]
        if len(companions)>1:raise FileAccessError('伴随时间戳文件不明确。')
        companion=self.provider.read(companions[0]['id'])[0] if companions else b''
        if len(companion)>2_000_000:raise FileAccessError('伴随时间戳超过 2 MB。')
        from urllib.error import HTTPError
        try:converted=self.service.binary('/api/v1/jobs/local-lip-conversion','POST',struct.pack('<II',len(raw),len(companion))+raw+companion,max_bytes=2_000_000)
        except HTTPError as error:
            code=json.load(error).get('detail','')
            raise FileAccessError({'legacy_conversion_budget':'旧 PKL 超过转换预算（16 MB、数值或内存上限）。',
                'legacy_conversion_timeout':'旧 PKL 转换超时，请缩短数据。','preview_busy':'预览或转换正在进行，请稍后重试。'}.get(code,'旧 PKL 含不支持或损坏的结构，未生成文件。')) from None
        with pin_directory(root):
            self.provider.directory(entry.directory)
            # Recheck sources after conversion; no file output if the selected input changed.
            if self.provider.read(file_id)[0]!=raw:raise FileAccessError('旧 PKL 已变化，请刷新。')
            if companions and self.provider.read(companions[0]['id'])[0]!=companion:raise FileAccessError('伴随时间戳已变化。')
            if target.exists() or target.is_symlink():raise FileAccessError('已有同名 .lip.json，保持原样；可直接关联它。')
            temp=root/('.ptb-'+uuid4().hex+'.part');original=None
            try:
                with temp.open('xb') as f:
                    original=identity(os.fstat(f.fileno()));f.write(converted);f.flush();os.fsync(f.fileno())
                os.rename(temp,target)
            finally:
                if original and temp.exists() and identity(temp.stat())==original:checked_path(temp);temp.unlink()
            file=next(f for f in self.provider.list(entry.directory) if f['name']==name)
            return {'file':file,'companion_found':bool(companions)}

    def save(self,batch_id,directory_id,*,single=False):
        directory=self.provider.directory(directory_id)
        if directory.purpose not in ('input','output'):raise FileAccessError('请选择结果目录。')
        root=directory.path
        if single:
            job=self.service.get('/api/v1/jobs/'+batch_id)
            if job['operation'] not in ('spectrogram_to_audio','egg_analysis','lpc_analysis') or job['state']!='succeeded':raise FileAccessError('分析结果尚不可用。')
            batch={'summary':{'items':[dict(state='succeeded',job_id=batch_id,index=0)]}}
        else:batch=self.service.get('/api/v1/jobs/batches/'+batch_id)
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
                    export_names={}
                    if job['operation'] in ('egg_analysis','lpc_analysis'):
                        import base64
                        meta=next(f for f in job['result_manifest']['files'] if f['name']==('lpc.ptb.json' if job['operation']=='lpc_analysis' else 'egg.ptb.json'))
                        verified=self.invoke(dict(op='result',job=job['id'],id=meta['id']))
                        export_names=json.loads(base64.b64decode(verified['base64'])).get('export_names',{})
                    for file in job['result_manifest']['files']:
                        name=file['name']
                        if job['operation'] in ('egg_analysis','lpc_analysis'):name=export_names.get(name,name)
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
