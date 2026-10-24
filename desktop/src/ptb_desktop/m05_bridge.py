"""M05 local transport, streaming disk saves and explicit directory capabilities."""
import base64
import hashlib
import json
import math
import os
import shutil
import time
import tempfile
import threading
from pathlib import Path
from uuid import UUID,uuid4
from .file_provider import FileAccessError,checked_path

def _scratch_directory(prefix):
    # macOS /var is a system alias. Resolve only our own temporary root;
    # user-selected paths still pass the unchanged capability checks.
    root=Path(tempfile.gettempdir()).resolve(strict=True)
    return tempfile.TemporaryDirectory(prefix=prefix,dir=root)

class M05Bridge:
    def __init__(self,bridge):
        from .m05_replay import ReplayReader
        self.bridge=bridge;self.writes={};self.replay=ReplayReader();self.recordings={};self.exports={};self.metadata={};self.playbacks={}
    def invoke(self,body):
        from urllib.error import HTTPError
        try:return self._invoke(body)
        except HTTPError as error:
            try:code=json.loads(error.read(65536)).get('detail','m05_execution_failed')
            except Exception:code='m05_execution_failed'
            raise FileAccessError(code if isinstance(code,str) else 'm05_execution_failed') from None
    def _invoke(self,body):
        service=self.bridge.service;op=body['op']
        if op=='m05_inspect_begin':
            size=body.get('size')
            if type(size) is not int or not 0<size<=128_000_000 or len(self.writes)>=8:raise FileAccessError('媒体检查超出预算。')
            scratch=_scratch_directory('ptb-m05-preview-');source=Path(scratch.name)/'source.media';source.open('xb').close()
            key=uuid4().hex;self.writes[key]=dict(path=source,size=size,written=0,hash=hashlib.sha256(),scratch=scratch,inspect=True)
            return dict(id=key)
        if op=='m05_audio_preview':
            job=service.get('/api/v1/jobs/'+str(UUID(body['job'])))
            if job['state']!='succeeded' or job['operation']!='lip_analysis':raise FileAccessError('唇形结果尚不可用。')
            item=next((f for f in job['result_manifest']['files'] if f['name']=='audio_recording.wav'),None)
            if not item or item['size_bytes']>128_000_000:raise FileAccessError('音频不存在或超出波形检查预算。')
            with _scratch_directory('ptb-m05-wave-') as scratch:
                source=Path(scratch)/'audio.wav'
                with source.open('xb') as f:
                    for pos in range(0,item['size_bytes'],262144):f.write(service.binary(f'/api/v1/jobs/local-results/{item["id"]}?offset={pos}&size={min(262144,item["size_bytes"]-pos)}'))
                if self.file_hash(source)!=item['sha256']:raise FileAccessError('音频检查校验失败。')
                return self.export_recording(source,action='inspect')
        if op=='m05_catalog':return service.get('/api/v1/jobs/m05/catalog')
        if op=='m05_replay':
            job=service.get('/api/v1/jobs/'+str(UUID(body['job'])))
            return self.replay.read(service,job,body['start'])
        if op=='m05_recording_input':
            item=self.recordings.get(body.get('token'))
            if not item:raise FileAccessError('请先保存本次录制。')
            source,sha=item;checked_path(source)
            if self.file_hash(source)!=sha:raise FileAccessError('已保存的 MP4 发生变化，请重新选择该视频。')
            upload=service.request('/api/v1/jobs/m05/uploads','POST',dict(name='raw_recording.mp4',size=source.stat().st_size))
            try:
                with source.open('rb') as stream:
                    offset=0
                    while raw:=stream.read(262144):
                        service.request('/api/v1/jobs/m05/uploads/'+upload['id'],'PUT',dict(offset=offset,base64=base64.b64encode(raw).decode()))
                        offset+=len(raw)
                return service.request('/api/v1/jobs/m05/uploads/'+upload['id']+'/finalize','POST')
            except Exception:
                service.request('/api/v1/jobs/m05/uploads/'+upload['id']+'/abort','POST')
                raise
        if op=='m05_history':
            jobs=[j for j in service.get('/api/v1/jobs?project_id=00000000-0000-4000-8000-000000000001')['jobs'] if j['operation']=='lip_analysis' and j['state']=='succeeded'][:100]
            from urllib.error import HTTPError
            history=[]
            for job in jobs:
                try:name=self.result_metadata(job).get('input_name') or ('视频 '+job['id'][:8])
                except (FileAccessError,HTTPError):continue
                history.append(dict(id=job['id'],name=name))
            return history
        if op=='m05_playback_info':
            job=self.checked_job(body['job']);source=self.result_metadata(job).get('input_asset')
            if not source:raise FileAccessError('此历史结果缺少原视频，请重新选择文件分析。')
            if len(self.playbacks)>=2:raise FileAccessError('正在准备其他回放，请稍后重试。')
            if shutil.disk_usage(tempfile.gettempdir()).free<650_000_000:raise FileAccessError('回放转换磁盘空间不足。')
            scratch=_scratch_directory('ptb-m05-playback-')
            try:
                path=Path(scratch.name)/'source.media';self.download_file(source,path)
                info=self.export_recording(path,action='playback')
                key=uuid4().hex;self.playbacks[key]=dict(scratch=scratch,path=path.parent/info['name'],info=info)
                return dict(token=key,**info)
            except Exception:
                scratch.cleanup();raise
        if op=='m05_playback_release':
            item=self.playbacks.pop(body.get('token'),None)
            if item:item['scratch'].cleanup()
            return dict(released=item is not None)
        if op=='m05_playback_block':
            item=self.playbacks.get(body.get('token'));offset=body.get('offset')
            if not item or type(offset) is not int or not 0<=offset<item['info']['size_bytes']:raise FileAccessError('回放视频读取范围不正确。')
            with item['path'].open('rb') as stream:stream.seek(offset);raw=stream.read(min(262144,item['info']['size_bytes']-offset))
            return dict(base64=base64.b64encode(raw).decode())
        if op in ('m05_source_info','m05_source_block'):
            job=self.checked_job(body['job']);source=self.result_metadata(job).get('input_asset')
            if not source:raise FileAccessError('此历史结果缺少原视频，请重新选择文件分析。')
            if op=='m05_source_info':return source
            offset=body.get('offset')
            if type(offset) is not int or not 0<=offset<source['size_bytes']:raise FileAccessError('视频读取范围不正确。')
            raw=service.binary(f'/api/v1/jobs/local-results/{str(UUID(source["id"]))}?offset={offset}&size={min(262144,source["size_bytes"]-offset)}')
            return dict(base64=base64.b64encode(raw).decode())
        if op=='m05_cancel_export':
            key=str(UUID(body['export_id']));event=self.exports.get(key)
            if event:event.set()
            return dict(cancelled=event is not None)
        if op=='m05_export_animation':
            job=self.checked_job(body['job']);directory=self.bridge.provider.directory(body['directory'])
            if directory.purpose!='output':raise FileAccessError('请选择输出目录。')
            offset=self.checked_offset(body['offset']);format=body['format'];quality=body['quality']
            if format not in ('mp4','gif') or quality not in ('high','standard','small'):raise FileAccessError('动画设置不正确。')
            name='lip-animation-'+job['id'][:8]+'-'+uuid4().hex[:8]+'.'+format
            key=str(UUID(body['export_id']));event=threading.Event()
            if len(self.exports)>=4:raise FileAccessError('正在导出其他动画，请稍后重试。')
            self.exports[key]=event
            try:self.render_result(job,directory.path,dict(video=False,animation=True,quality=quality,format=format),offset,animation_name=name,cancel=event)
            finally:self.exports.pop(key,None)
            return dict(saved=True,file=name)
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
            offset=self.checked_offset(body['offset']);action=body['action']
            options=body.get('options',dict(video=False,animation=False,quality='high'))
            if set(options)!={'video','animation','quality'} or any(type(options[k]) is not bool for k in ('video','animation')) or options['quality'] not in ('high','standard','small'):raise FileAccessError('结果保存选项不正确。')
            if action not in ('apply','save_without_offset') or not isinstance(offset,(float,int)) or not math.isfinite(offset) or abs(offset)>2:raise FileAccessError('偏移参数不正确。')
            target=directory.path/('M05-'+job['id'][:8]+'-'+uuid4().hex[:8]);checked_path(directory.path);target.mkdir()
            for item in job['result_manifest']['files']:
                name=item['name']
                if 'options' in body and name in ('lip-animation.mp4','lip-animation.gif'):continue
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
            media=self.render_result(job,target,options,applied,base_bytes=sum(p.stat().st_size for p in target.iterdir() if p.is_file())) if options['video'] or options['animation'] else None
            alignment=dict(schema='m05-alignment/1',lip_manual_offset=applied,apply='time_s + lip_manual_offset exactly once',parent_job=job['id'])
            exchange=target/'audio_recording.lip.json'
            if exchange.exists():
                value=json.loads(exchange.read_text('utf8'));value['data']['metadata']['lip_manual_offset']=applied
                # The companion picked by M01/M12 must carry the chosen offset.
                exchange.rename(target/'unaligned.lip.json')
                self.write_json(exchange,value)
                self.write_json(target/'aligned.lip.json',value)
            self.write_json(target/'alignment.json',alignment)
            self.write_json(target/'saved-manifest.json',dict(schema='m05-saved/1',parent_job=job['id'],lip_manual_offset=applied,
                source_manifest='manifest.json',source_exchange='unaligned.lip.json',save_options=options,media=media,files=[dict(name=p.name,bytes=p.stat().st_size,sha256=self.file_hash(p)) for p in target.iterdir() if p.is_file()]))
            return dict(saved=True,directory=target.name)
        if op=='m05_save_begin':
            directory=self.bridge.provider.directory(body['directory']);name=body['name'];size=body['size']
            if directory.purpose!='output' or not isinstance(name,str) or Path(name).name!=name or any(c in name for c in '/\\:\x00') or not name.endswith(('.webm','.mp4','.json')) or not isinstance(size,int) or not 0<size<=128_000_000:raise FileAccessError('录制保存参数不正确。')
            if len(self.writes)>=8:raise FileAccessError('存在过多未完成保存，请保留当前录制并检查磁盘。')
            checked_path(directory.path)
            if shutil.disk_usage(directory.path).free<size+(600_000_000 if body.get('recording') else 64_000_000):raise FileAccessError('所选磁盘可用空间不足，录制仍保留在内存。')
            if body.get('recording') and shutil.disk_usage(tempfile.gettempdir()).free<size+650_000_000:raise FileAccessError('临时磁盘空间不足，录制仍保留在内存。')
            key=uuid4().hex;target=directory.path/('M05-'+key[:8]+'-'+name)
            metadata_id=None
            if body.get('recording') is True:
                metadata_size=body.get('metadata_size')
                if type(metadata_size) is not int or not 0<metadata_size<=36_000_000:raise FileAccessError('采集记录超过保存预算。')
                destination=directory.path/('M05-recording-'+key[:8]);destination.mkdir()
                scratch=_scratch_directory('ptb-m05-save-');target=Path(scratch.name)
                metadata_id=uuid4().hex;meta=target/'capture.m05-preview.json';meta.open('xb').close()
                self.writes[metadata_id]=dict(path=meta,size=metadata_size,written=0,hash=hashlib.sha256())
                target=target/'.pending-media'
            target.open('xb').close();self.writes[key]=dict(path=target,size=size,written=0,hash=hashlib.sha256())
            if metadata_id:self.writes[key].update(metadata=self.writes[metadata_id],scratch=scratch,destination=destination)
            return dict(id=key,metadata_id=metadata_id)
        if op=='m05_save_abort':
            # Keep published partial output; release only this operation-owned scratch.
            item=self.writes.pop(body['id'],None)
            if item and 'scratch' in item:item['scratch'].cleanup()
            return dict(aborted=True)
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
            if item.get('inspect'):
                try:return dict(saved=False,inspection=self.export_recording(item['path'],action='inspect'))
                finally:item['scratch'].cleanup();self.writes.pop(body['id'],None)
            if 'metadata' in item:
                meta=item['metadata']
                if meta['written']!=meta['size'] or self.file_hash(meta['path'])!=meta['hash'].hexdigest():raise FileAccessError('候选记录保存不完整。')
                info=self.export_recording(item['path'],destination=item['destination'])
                if len(self.recordings)>=8:self.recordings.pop(next(iter(self.recordings)))
                media=item['destination']/'raw_recording.mp4'
                if media.is_file():self.recordings[body['id']]=(media,self.file_hash(media));info['token']=body['id']
                item['scratch'].cleanup()
                del self.writes[body['id']]
                return dict(saved=True,recording=info)
            del self.writes[body['id']];return dict(saved=True)
        raise FileAccessError('唇形提取不支持此操作。')
    def checked_job(self,key):
        job=self.bridge.service.get('/api/v1/jobs/'+str(UUID(key)))
        if job['operation']!='lip_analysis' or job['state']!='succeeded':raise FileAccessError('唇形完整结果尚不可用。')
        return job
    def result_metadata(self,job):
        item=next((f for f in job['result_manifest']['files'] if f['name']=='manifest.json'),None)
        if not item or not 0<item['size_bytes']<=16_000_000:raise FileAccessError('结果信息不完整。')
        key=(job['id'],item['sha256'])
        if key in self.metadata:return self.metadata[key]
        raw=bytearray()
        for offset in range(0,item['size_bytes'],262144):raw.extend(self.bridge.service.binary(f'/api/v1/jobs/local-results/{item["id"]}?offset={offset}&size={min(262144,item["size_bytes"]-offset)}'))
        if len(raw)!=item['size_bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise FileAccessError('结果信息校验失败。')
        value=json.loads(raw)
        if len(self.metadata)>=100:self.metadata.pop(next(iter(self.metadata)))
        self.metadata[key]=value
        return value
    def download_file(self,item,target):
        if type(item['size_bytes']) is not int or not 0<item['size_bytes']<=512_000_000:raise FileAccessError('媒体读取超出预算。')
        checked_path(target.parent);h=hashlib.sha256();size=0
        with target.open('xb') as dest:
            for offset in range(0,item['size_bytes'],262144):
                raw=self.bridge.service.binary(f'/api/v1/jobs/local-results/{str(UUID(item["id"]))}?offset={offset}&size={min(262144,item["size_bytes"]-offset)}')
                if dest.write(raw)!=len(raw):raise OSError('short_write')
                h.update(raw);size+=len(raw)
        if size!=item['size_bytes'] or h.hexdigest()!=item['sha256']:raise FileAccessError('媒体读取校验失败。')
    def render_result(self,job,target,options,offset,*,animation_name='lip-animation.mp4',cancel=None,base_bytes=0):
        checked_path(target)
        if shutil.disk_usage(target).free<650_000_000 or shutil.disk_usage(tempfile.gettempdir()).free<1_000_000_000:raise FileAccessError('媒体导出磁盘空间不足。')
        with _scratch_directory('ptb-m05-result-') as scratch:
            stage=Path(scratch)
            needed={'manifest.json'}|({'frames.jsonl','audio_recording.wav'} if options['animation'] else set())
            for name in needed:
                item=next((f for f in job['result_manifest']['files'] if f['name']==name),None)
                if item is None:
                    if name=='audio_recording.wav':continue
                    raise FileAccessError('动画所需结果不完整。')
                self.download_file(item,stage/name)
            source=None
            if options['video']:
                metadata=json.loads((stage/'manifest.json').read_text('utf8'));item=metadata.get('input_asset')
                if not item:raise FileAccessError('此历史结果缺少原视频，请重新选择文件分析后保存视频。')
                suffix=Path(item['name']).suffix.lower()
                if suffix not in ('.mp4','.mov','.avi','.mkv','.wmv','.m4v','.webm'):raise FileAccessError('视频类型不正确。')
                source=stage/('source'+suffix);self.download_file(item,source)
            return self.export_recording(stage/'manifest.json',destination=target,action='render_result',cancel=cancel,options=options,offset=offset,source_video=str(source) if source else None,animation_name=animation_name,base_bytes=base_bytes)
    @staticmethod
    def checked_offset(value):
        if type(value) not in (int,float) or not math.isfinite(value) or abs(value)>2:raise FileAccessError('偏移参数不正确。')
        return value
    @staticmethod
    def file_hash(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    @staticmethod
    def export_recording(source,*,destination=None,action='export',cancel=None,**options):
        import sys
        if sys.platform=='darwin':
            from ptb_worker.native.darwin import OwnedProcess
        else:
            from ptb_worker.native.windows import OwnedProcess
        from ptb_worker import m05_media_export
        runtime=sys.executable if sys.platform=='darwin' else os.environ.get('PTB_M05_PYTHON')
        if not runtime or not Path(runtime).is_file():raise FileAccessError('视频编码环境不可用，原录制仍保留。')
        root=source.parent;request=root/'export-request.json'
        M05Bridge.write_json(request,dict(source=str(source),action=action,**options,**(dict(destination=str(destination)) if destination else {})))
        process=None;started=time.monotonic()
        try:
            process=OwnedProcess([runtime,'-I','-B',str(Path(m05_media_export.__file__)),str(request)],root,1_073_741_824)
            while process.poll() is None:
                if cancel is not None and cancel.is_set():raise FileAccessError('动画导出已取消，分析结果保留。')
                if time.monotonic()-started>620:raise FileAccessError('音视频回放转换超时，请重试。' if action=='playback' else 'MP4 保存超时，原录制仍保留。')
                time.sleep(.05)
            response=root/'export-response.json'
            if process.poll()!=0 or not response.is_file():raise FileAccessError('音视频回放转换未完成，请重试。' if action=='playback' else 'MP4 编码未完成，原录制仍保留。')
            value=json.loads(response.read_text('utf8'))
            if not value.get('ok'):raise FileAccessError(value.get('error','MP4 保存失败。'))
            # Source is a uniquely created scratch file owned by this save operation.
            if action=='export':source.unlink()
            return value['value']
        finally:
            if process:process.close()
    @staticmethod
    def write_json(path,value):
        with path.open('x',encoding='utf8') as stream:
            json.dump(value,stream,ensure_ascii=False,allow_nan=False);stream.flush();os.fsync(stream.fileno())
