"""M08 desktop transport to the same durable HTTP service, no scientific work."""
import base64
import hashlib
import os
from pathlib import Path
import re
from uuid import UUID
from .file_provider import FileAccessError,checked_path,identity
from .task_bridge import PROJECT,pin_directory


class M08Bridge:
    def __init__(self,bridge):
        self.bridge=bridge;self.provider=bridge.provider;self.service=bridge.service
        self.exports={}

    def invoke(self,body):
        op=body['op']
        if op=='m08_source':
            raw,sha=self.provider.read(body['id']);entry=self.provider.entries[body['id']]
            if body.get('sha256') and body['sha256']!=sha:raise FileAccessError('源音频已变化，请刷新。')
            return self.service.import_input(raw,entry.name,'audio')
        if op=='m08_call':
            action=body['action']
            if action=='list':return self.service.get('/api/v1/jobs/m08/list/'+PROJECT)
            if action not in ('create','save','history','remove','rename'):raise FileAccessError('M08 操作不可用。')
            value=body['body']
            if value.get('project_id')!=PROJECT:raise FileAccessError('项目不匹配。')
            # External exports remain explicit snapshots. Before modifying an exported
            # result, require its originally granted directory and exact current hash.
            ids=value.get('ids',[])
            selected=[(i,record) for i in ids for record in self.exports.get(i,[])]
            for _,record in selected:self._verify(record)
            if action=='rename':
                for result_id,record in selected:
                    name=value['names'][ids.index(result_id)]
                    if not name or any(c in name for c in '/\\:<>"|?*') or len(name)>180:raise FileAccessError('无效文件名。')
                    destination=record['path'].parent/name
                    if destination!=record['path'] and destination.exists():raise FileAccessError('文件名冲突，原文件保留。')
            answer=self.service.request('/api/v1/jobs/m08/'+action,'POST',value)
            if action in ('remove','rename'):
                for result_id,record in selected:
                    self._verify(record)
                    try:
                        if action=='remove' and result_id in answer['removed']:
                            record['path'].unlink();self.exports[result_id].remove(record)
                        elif action=='rename':
                            destination=record['path'].parent/value['names'][ids.index(result_id)]
                            if destination!=record['path']:os.rename(record['path'],destination)
                            record['path']=destination
                    except OSError as exc:
                        raise FileAccessError('受管结果已更新，但本地导出副本操作失败，请核对输出目录。') from exc
            return answer
        if op=='m08_export':return self.export(body)
        raise FileAccessError('M08 操作不可用。')

    def _verify(self,record):
        self.provider.directory(record['directory']);path=checked_path(record['path'])
        if identity(path.stat())!=record['identity'] or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise FileAccessError('本地结果已由外部修改，停止管理。')

    def export(self,body):
        directory=self.provider.directory(body['directory'])
        if directory.purpose!='output':raise FileAccessError('需要输出目录授权。')
        job=self.service.get('/api/v1/jobs/'+str(UUID(body['job'])))
        if job['operation']!='pitch_manipulation' or job['state']!='succeeded':raise FileAccessError('结果未完成。')
        item=next((f for f in job['result_manifest']['files'] if f['id']==body['id'] and f['name'].endswith('.wav')),None)
        if not item or item['id'] in job['result_manifest'].get('deleted',[]):raise FileAccessError('结果不可用。')
        direct=body.get('direct') is True
        name=job['result_manifest'].get('aliases',{}).get(item['id'],item['name'])
        if not name or any(ord(c)<32 or c in '/\\:<>"|?*' for c in name):raise FileAccessError('无效结果名。')
        verified=self.bridge.invoke(dict(op='result',job=job['id'],id=item['id']))
        raw=base64.b64decode(verified['base64']);root=directory.path
        if direct:
            for record in self.exports.get(item['id'],[]):
                if record['path'].parent==root:
                    self._verify(record)
                    self.bridge.record_export([item['id']])
                    return dict(name=record['path'].name,id=item['id'],directory=str(root))
        with pin_directory(root):
            self.provider.directory(body['directory'])
            # Exclusive creation is the final cross-process allocation gate.
            match=None if direct else re.match(r'^(.*_modified_)(\d+)\.wav$',name)
            base_name=name
            for index in range(10000):
                if direct and index:
                    suffix=f' ({index+1}).wav'
                    name=Path(base_name).stem[:180-len(suffix)]+suffix
                if match:
                    numbers=[int(m.group(1)) for p in root.iterdir() if (m:=re.fullmatch(re.escape(match.group(1))+r'(\d+)\.wav',p.name,re.I))]
                    name=match.group(1)+str(max(numbers,default=0)+1)+'.wav'
                path=root/name
                created=None
                try:
                    with path.open('xb') as stream:
                        created=identity(os.fstat(stream.fileno()))
                        stream.write(raw);stream.flush();os.fsync(stream.fileno())
                    break
                except FileExistsError:
                    if not match and not direct:raise FileAccessError('结果文件名已存在。') from None
                except BaseException:
                    if created and path.exists() and identity(path.stat())==created:path.unlink()
                    raise
            else:raise FileAccessError('无法分配结果编号。')
            self.exports.setdefault(item['id'],[]).append(dict(path=path,directory=body['directory'],identity=identity(path.stat()),sha256=item['sha256']))
            answer=dict(name=name,id=item['id'],directory=str(root))
            warning=self.bridge.record_export([item['id']])
            if warning:answer['retention_warning']=warning
            return answer
