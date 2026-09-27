"""M06 local file handles and the existing durable host. No scientific work."""
from .file_provider import FileAccessError
from .task_bridge import PROJECT


class M06Bridge:
    def __init__(self,bridge):self.bridge=bridge

    def invoke(self,body):
        if body['op']=='m06_parameters':
            raw=body['text'].encode('utf8')
            if not 0<len(raw)<=2_000_000:raise FileAccessError('完整参数超过 2 MB')
            return self.bridge.service.import_input(raw,'m06-parameters.csv','table')
        if body['op']=='m06_source':
            raw,sha=self.bridge.provider.read(body['id']);entry=self.bridge.provider.entries[body['id']]
            if len(raw)>8_000_000 or not entry.name.lower().endswith('.wav'):raise FileAccessError('M06 仅接受 8 MB 内 WAV')
            if body.get('sha256') and body['sha256']!=sha:raise FileAccessError('源音频已变化')
            return self.bridge.service.import_input(raw,entry.name,'audio')
        if body['op']=='m06_create':
            if body['body'].get('project_id')!=PROJECT:raise FileAccessError('项目不匹配')
            return self.bridge.service.request('/api/v1/jobs/m06/create','POST',body['body'])
        raise FileAccessError('M06 操作不可用')
