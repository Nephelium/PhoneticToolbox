"""M05 local transport, streaming disk saves and explicit directory capabilities."""
import base64
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from uuid import UUID,uuid4
from .file_provider import FileAccessError,checked_path

class M05Bridge:
    def __init__(self,bridge):self.bridge=bridge;self.writes={}
    def invoke(self,body):
        from urllib.error import HTTPError
        try:return self._invoke(body)
        except HTTPError as error:
            try:code=json.loads(error.read(65536)).get('detail','m05_execution_failed')
            except Exception:code='m05_execution_failed'
            raise FileAccessError(code if isinstance(code,str) else 'm05_execution_failed') from None
    def _invoke(self,body):
        service=self.bridge.service;op=body['op']
        if op=='m05_catalog':return service.get('/api/v1/jobs/m05/catalog')
        if op=='m05_history':return [j for j in service.get('/api/v1/jobs?project_id=00000000-0000-4000-8000-000000000001')['jobs'] if j['operation']=='lip_analysis' and j['state']=='succeeded'][:100]
        if op=='m05_repeat':return service.request('/api/v1/jobs/m05/'+str(UUID(body['job']))+'/repeat','POST',body['config'])
        if op=='m05_create':return service.request('/api/v1/jobs/m05/create','POST',body['body'])
        if op=='m05_upload_begin':return service.request('/api/v1/jobs/m05/uploads','POST',dict(name=body['name'],size=body['size']))
        if op=='m05_upload_block':return service.request('/api/v1/jobs/m05/uploads/'+str(UUID(body['id'])),'PUT',dict(offset=body['offset'],base64=body['base64']))
        if op=='m05_upload_finish':return service.request('/api/v1/jobs/m05/uploads/'+str(UUID(body['id']))+'/finalize','POST')
        if op=='m05_upload_abort':return service.request('/api/v1/jobs/m05/uploads/'+str(UUID(body['id']))+'/abort','POST')
        if op=='m05_save':
            job=service.get('/api/v1/jobs/'+str(UUID(body['job'])))
            if job['state']!='succeeded' or job['operation']!='lip_analysis':raise FileAccessError('唇形完整结果尚不可用。')
            directory=self.bridge.provider.directory(body['directory'])
            if directory.purpose!='output':raise FileAccessError('请选择输出目录。')
            offset=body['offset'];action=body['action']
            if action not in ('apply','save_without_offset') or not isinstance(offset,(float,int)) or not math.isfinite(offset) or abs(offset)>2:raise FileAccessError('偏移参数不正确。')
            target=directory.path/('M05-'+job['id'][:8]+'-'+uuid4().hex[:8]);checked_path(directory.path);target.mkdir()
            for item in job['result_manifest']['files']:
                name=item['name']
                if Path(name).name!=name or any(c in name for c in '/\\:\x00'):raise FileAccessError('结果名不安全。')
                checked_path(target);h=hashlib.sha256();size=0
                with (target/name).open('xb') as stream:
                    for pos in range(0,item['size_bytes'],262144):
                        raw=service.binary(f'/api/v1/jobs/local-results/{item["id"]}?offset={pos}&size={min(262144,item["size_bytes"]-pos)}')
                        if stream.write(raw)!=len(raw):raise OSError('short_write')
                        h.update(raw);size+=len(raw)
                    stream.flush();os.fsync(stream.fileno())
                if size!=item['size_bytes'] or h.hexdigest()!=item['sha256']:raise FileAccessError('保存校验失败，部分结果已保留。')
            applied=offset if action=='apply' else 0.
            alignment=dict(schema='m05-alignment/1',lip_manual_offset=applied,apply='time_s + lip_manual_offset exactly once',parent_job=job['id'])
            exchange=target/'audio_recording.lip.json'
            if exchange.exists():
                value=json.loads(exchange.read_text('utf8'));value['data']['metadata']['lip_manual_offset']=applied
                self.write_json(target/'aligned.lip.json',value)
            self.write_json(target/'alignment.json',alignment)
            return dict(saved=True,directory=target.name)
        if op=='m05_save_begin':
            directory=self.bridge.provider.directory(body['directory']);name=body['name'];size=body['size']
            if directory.purpose!='output' or not isinstance(name,str) or Path(name).name!=name or any(c in name for c in '/\\:\x00') or not name.endswith(('.webm','.json')) or not isinstance(size,int) or not 0<size<=128_000_000:raise FileAccessError('录制保存参数不正确。')
            if len(self.writes)>=8:raise FileAccessError('存在过多未完成保存，请保留当前录制并检查磁盘。')
            checked_path(directory.path)
            if shutil.disk_usage(directory.path).free<size+64_000_000:raise FileAccessError('所选磁盘可用空间不足，录制仍保留在内存。')
            key=uuid4().hex;target=directory.path/('M05-'+key[:8]+'-'+name)
            target.open('xb').close();self.writes[key]=dict(path=target,size=size,written=0,hash=hashlib.sha256())
            return dict(id=key)
        if op=='m05_save_block':
            item=self.writes.get(body['id'])
            if item is None:raise FileAccessError('保存会话已失效。')
            raw=base64.b64decode(body['base64'],validate=True)
            if not 0<len(raw)<=262144 or body['offset']!=item['written'] or item['written']+len(raw)>item['size']:raise FileAccessError('保存块不正确。')
            checked_path(item['path'])
            with item['path'].open('ab') as stream:
                if stream.tell()!=item['written']:raise FileAccessError('保存文件发生变化。')
                if stream.write(raw)!=len(raw):raise OSError('short_write')
                stream.flush();os.fsync(stream.fileno())
            item['written']+=len(raw);item['hash'].update(raw);return dict(bytes=item['written'])
        if op=='m05_save_finish':
            item=self.writes.get(body['id'])
            if item is None or item['written']!=item['size']:raise FileAccessError('录制尚未完整保存。')
            h=hashlib.sha256();checked_path(item['path'])
            with item['path'].open('rb') as stream:
                for raw in iter(lambda:stream.read(262144),b''):h.update(raw)
            if h.hexdigest()!=item['hash'].hexdigest():raise FileAccessError('录制保存后校验失败。')
            del self.writes[body['id']];return dict(saved=True)
        raise FileAccessError('M05 操作不受支持。')
    @staticmethod
    def write_json(path,value):
        with path.open('x',encoding='utf8') as stream:
            json.dump(value,stream,ensure_ascii=False,allow_nan=False);stream.flush();os.fsync(stream.fileno())
