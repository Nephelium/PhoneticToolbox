"""M07 uses authorized local file handles and shared tasks, no algorithms."""
from .file_provider import FileAccessError
from .task_bridge import PROJECT

class M07Bridge:
    def __init__(self,bridge):self.bridge=bridge
    def invoke(self,body):
        if body['op']=='m07_source':
            raw,sha=self.bridge.provider.read(body['id']);entry=self.bridge.provider.entries[body['id']]
            if len(raw)>8_000_000 or not entry.name.lower().endswith('.wav'):raise FileAccessError('发声类型合成支持 8 MB 以内的 WAV 文件')
            if body.get('sha256') and body['sha256']!=sha:raise FileAccessError('输入音频已变化')
            return self.bridge.service.import_input(raw,entry.name,'audio')
        if body['op']=='m07_create':
            if body['body'].get('project_id')!=PROJECT:raise FileAccessError('项目不匹配')
            return self.bridge.service.request('/api/v1/jobs/m07/create','POST',body['body'])
        if body['op']=='m07_list':
            result=self.bridge.service.get('/api/v1/jobs?project_id='+PROJECT)
            return [j for j in result['jobs'] if j['operation']=='phonation_synthesis']
        if body['op']=='m07_f0':
            import base64
            import json
            from uuid import UUID
            from phonetic_core.manipulation.m07_display import synthesis_f0_display
            job_id=str(UUID(body['job']));job=self.bridge.service.get('/api/v1/jobs/'+job_id)
            if job['operation']!='phonation_synthesis' or job['state']!='succeeded':raise FileAccessError('本组尚未完整成功')
            meta=next(f for f in job['result_manifest']['files'] if f['name']=='m07.ptb.json')
            if meta['size_bytes']>2_000_000:raise FileAccessError('基频曲线数据过多，请缩小分析范围')
            verified=self.bridge.invoke(dict(op='result',job=job_id,id=meta['id']))
            return synthesis_f0_display(json.loads(base64.b64decode(verified['base64'])))
        if body['op']=='m07_save':
            from uuid import UUID
            from .task_bridge import pin_directory
            from .file_provider import checked_path
            job_id=str(UUID(body['job']));job=self.bridge.service.get('/api/v1/jobs/'+job_id)
            if job['operation']!='phonation_synthesis' or job['state']!='succeeded':raise FileAccessError('本组尚未完整成功')
            directory=self.bridge.provider.directory(body['directory'])
            if directory.purpose!='output':raise FileAccessError('请选择输出位置')
            name=body.get('name')
            if not isinstance(name,str) or not name or len(name)>120 or name[-1] in ' .' or any(ord(c)<32 or c in '/\\:<>"|?*' for c in name):
                raise FileAccessError('结果组名称不正确，请重新选择本组保存')
            with pin_directory(directory.path):
                self.bridge.provider.directory(body['directory'])
                for number in range(1,10001):
                    destination=directory.path/(name if number==1 else f'{name}（{number}）')
                    try:destination.mkdir()
                    except FileExistsError:continue
                    checked_path(destination)
                    break
                else:raise FileAccessError('同名结果过多，请选择新的输出位置')
                grant=self.bridge.provider.choose('output',lambda:str(destination))
                return self.bridge.save(job_id,grant['id'],single=True)
        raise FileAccessError('发声类型合成操作不可用')
