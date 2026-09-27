"""M07 uses authorized local file handles and shared tasks, no algorithms."""
from .file_provider import FileAccessError
from .task_bridge import PROJECT

class M07Bridge:
    def __init__(self,bridge):self.bridge=bridge
    def invoke(self,body):
        if body['op']=='m07_source':
            raw,sha=self.bridge.provider.read(body['id']);entry=self.bridge.provider.entries[body['id']]
            if len(raw)>8_000_000 or not entry.name.lower().endswith('.wav'):raise FileAccessError('M07 仅接受 8 MB 内 WAV')
            if body.get('sha256') and body['sha256']!=sha:raise FileAccessError('输入音频已变化')
            return self.bridge.service.import_input(raw,entry.name,'audio')
        if body['op']=='m07_create':
            if body['body'].get('project_id')!=PROJECT:raise FileAccessError('项目不匹配')
            return self.bridge.service.request('/api/v1/jobs/m07/create','POST',body['body'])
        if body['op']=='m07_list':
            result=self.bridge.service.get('/api/v1/jobs?project_id='+PROJECT)
            return [j for j in result['jobs'] if j['operation']=='phonation_synthesis']
        if body['op']=='m07_save':
            from uuid import UUID
            from .task_bridge import pin_directory
            from .file_provider import checked_path
            job_id=str(UUID(body['job']));job=self.bridge.service.get('/api/v1/jobs/'+job_id)
            if job['operation']!='phonation_synthesis' or job['state']!='succeeded':raise FileAccessError('本组尚未完整成功')
            directory=self.bridge.provider.directory(body['directory'])
            if directory.purpose!='output':raise FileAccessError('请选择输出位置')
            with pin_directory(directory.path):
                self.bridge.provider.directory(body['directory'])
                destination=directory.path/('M07-'+job_id)
                destination.mkdir(exist_ok=True);checked_path(destination)
                grant=self.bridge.provider.choose('output',lambda:str(destination))
                return self.bridge.save(job_id,grant['id'],single=True)
        raise FileAccessError('M07 操作不可用')
