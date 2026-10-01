"""M11 explicit native file grants and the existing local task transport."""
import base64
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4,UUID
from .file_provider import FileAccessError,checked_path
from .task_requests import M11_INPUT_LIMITS


class M11Bridge:
    def __init__(self,bridge):
        self.bridge=bridge
        self.grants={}

    def grant(self,purpose,path):
        path=Path(path).absolute();checked_path(path)
        if purpose not in ('runtime','model','dictionary','archive','manifest','corpus'):raise FileAccessError('不支持的资源类型。')
        if not path.exists():raise FileAccessError('文件不存在。')
        key=uuid4().hex;self.grants[key]=(purpose,path)
        return dict(id=key,label=path.name,purpose=purpose)

    def path(self,key,purpose):
        if key not in self.grants or self.grants[key][0]!=purpose:raise FileAccessError('请重新选择对应资源。')
        path=self.grants[key][1];checked_path(path)
        return path

    def invoke(self,body):
        service=self.bridge.service
        op=body['op']
        if op=='m11_catalog':return service.get('/api/v1/jobs/m11/catalog')
        if op=='m11_jobs':return [j for j in service.get('/api/v1/jobs?project_id=00000000-0000-4000-8000-000000000001')['jobs'] if j['operation']=='mfa_alignment']
        if op=='m11_create':return service.request('/api/v1/jobs/m11/create','POST',body['body'])
        if op=='m11_component':
            values=body['body'];action=values['action']
            request=dict(action=action,model=str(self.path(values['model'],'model')),dictionary=str(self.path(values['dictionary'],'dictionary')))
            if action=='check':request['runtime']=str(self.path(values['runtime'],'runtime'))
            elif action=='import':request.update(archive=str(self.path(values['archive'],'archive')),manifest=str(self.path(values['manifest'],'manifest')),trusted_manifest_sha256=values['trusted_manifest_sha256'])
            else:raise FileAccessError('组件操作不受支持。')
            from urllib.error import HTTPError
            try:return service.request('/api/v1/jobs/m11/component','POST',request)
            except HTTPError as exc:raise FileAccessError(json.load(exc).get('detail','m11_component_failed')) from None
        if op=='m11_import':
            role=body.get('role');name=body.get('name')
            if role not in ('audio','transcript','dictionary'):raise FileAccessError('输入类型不正确。')
            encoded=body.get('base64','')
            limit=M11_INPUT_LIMITS[role]
            if not isinstance(encoded,str) or len(encoded)>((limit+2)//3)*4:raise FileAccessError('文件超出输入预算。')
            raw=base64.b64decode(encoded,validate=True)
            if len(raw)>limit:raise FileAccessError('文件超出输入预算。')
            return service.import_input(raw,name,role)
        if op=='m11_corpus':
            root=self.path(body['id'],'corpus')
            entries=list(root.rglob('*'))
            if len(entries)>10000:raise FileAccessError('目录条目超过 10,000，请选择具体语料目录。')
            planned=[];total=0
            for audio in sorted(p for p in entries if p.is_file() and p.suffix.lower()=='.wav'):
                checked_path(audio)
                candidates=[p for p in audio.parent.iterdir() if p.is_file() and p.stem==audio.stem and p.suffix.lower() in ('.lab','.txt','.textgrid')]
                if len(candidates)!=1:raise FileAccessError('每个 WAV 必须有且仅有一份同名 LAB、TXT 或 TextGrid 转写。')
                text=candidates[0];checked_path(text)
                if audio.stat().st_size>64_000_000 or not 0<text.stat().st_size<=2_000_000:raise FileAccessError('文件超过预算或转写为空。')
                total+=audio.stat().st_size+text.stat().st_size
                name=audio.relative_to(root).as_posix()
                if len(name.encode('utf8'))>110:raise FileAccessError('相对路径超过当前中文编码预算，请选择更具体的语料目录。')
                planned.append((audio,text,name))
                if len(planned)>100 or total>64_000_000:raise FileAccessError('单任务最多 100 份音频且总输入最多 64 MB。')
            if not planned:raise FileAccessError('目录中没有 WAV。')
            return [dict(name=name,audio=service.import_input(audio.read_bytes(),audio.name,'audio'),
                         transcript=service.import_input(text.read_bytes(),text.name,'transcript'),
                         transcript_format='.TextGrid' if text.suffix.lower()=='.textgrid' else text.suffix.lower())
                    for audio,text,name in planned]
        if op=='m11_log':
            job=service.get('/api/v1/jobs/'+str(UUID(body['id'])))
            if job['operation']!='mfa_alignment':raise FileAccessError('任务类型不正确。')
            from ptb_worker.mfa.runtime import registry_root,load_registry
            from ptb_worker.mfa.logs import read_log
            root=registry_root()/'attempts'
            attempts=sorted(root.glob(job['id']+'-'+str(job['generation'])+'-*')) if root.exists() else []
            if not attempts:return dict(text='',truncated=False)
            record=next((r for r in load_registry()['runtimes'] if r['id']==job['snapshot']['request']['runtime_id']),None) if 'snapshot' in job else None
            return read_log(attempts[-1],record['path'] if record else None)
        if op=='m11_events':return service.get('/api/v1/jobs/'+str(UUID(body['id']))+'/events')
        if op=='m11_save':
            job=service.get('/api/v1/jobs/'+str(UUID(body['job'])))
            if job['state']!='succeeded' or job['operation']!='mfa_alignment':raise FileAccessError('完整结果尚不可用。')
            directory=self.bridge.provider.directory(body['directory'])
            if directory.purpose!='output':raise FileAccessError('请选择输出目录。')
            # A new result subdirectory preserves input TextGrids even when the
            # chosen output directory equals the input directory.
            target=directory.path/('MFA-'+job['id'][:8]+'-'+uuid4().hex[:8])
            checked_path(directory.path);target.mkdir()
            saved=[]
            for item in job['result_manifest']['files']:
                name=item['name']
                if Path(name).name!=name or any(c in name for c in '/\\:\x00'):raise FileAccessError('结果名不安全。')
                value=self.bridge.invoke(dict(op='result',job=job['id'],id=item['id']))
                raw=base64.b64decode(value['base64'])
                checked_path(target)
                with (target/name).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
                if hashlib.sha256((target/name).read_bytes()).hexdigest()!=item['sha256']:raise FileAccessError('保存校验失败，已写入部分结果保留。')
                saved.append(name)
            return dict(count=len(saved),directory=target.name,saved=saved)
        raise FileAccessError('MFA 操作不受支持。')
