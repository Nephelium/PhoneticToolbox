"""Native capability to local API translation and non-overwriting result export."""
import hashlib
import json
import os
import struct
from contextlib import ExitStack
from pathlib import Path
from uuid import UUID,uuid4
from .file_provider import checked_path,FileAccessError,identity,open_locked
from .directory_io import pin_directory

PROJECT='00000000-0000-4000-8000-000000000001'


def parameter_export_paths(files,root,output,sha):
    """Keep a parameter/segment pair together, using human-readable suffixes."""
    groups={}
    for file in files:
        name=file['export_name']
        extension=next((suffix for suffix in ('.ptb.sqlite','.xlsx','.wav') if name.lower().endswith(suffix)),None)
        if extension is None:raise FileAccessError('参数结果文件格式不受支持。')
        base=name[:-len(extension)]
        groups.setdefault(base,[]).append((file,name[-len(extension):]))
    paths={}
    for base,group in groups.items():
        for number in range(1,10001):
            suffix='' if number==1 else f' ({number})'
            candidates=[(file,root/(base+suffix+extension)) for file,extension in group]
            if any(len(path.name)>220 for _,path in candidates):raise FileAccessError('输出文件名过长，请缩短音频名。')
            conflict=False
            for file,path in candidates:
                if output.exists(path):
                    checked_path(path)
                    if output.stat(path).st_size!=file['size_bytes'] or sha(path,output)!=file['sha256']:
                        conflict=True;break
            if conflict:continue
            paths.update((file['id'],path) for file,path in candidates)
            break
        else:raise FileAccessError('同名结果过多，请选择新的结果目录。')
    return paths


class TaskBridge:
    def __init__(self,provider,service):
        self.provider,self.service=provider,service
        self.batch_sources={}
        self.egg_preview_sources={}
        self.stream_sources={}
        from .annotation import AnnotationFiles
        self.annotation=AnnotationFiles(provider)
        from .m08_bridge import M08Bridge
        self.m08=M08Bridge(self)
        from .m14_bridge import M14Bridge
        self.m14=M14Bridge(self)
        from .m05_bridge import M05Bridge
        self.m05=M05Bridge(self)
        from .m11_bridge import M11Bridge
        self.m11=M11Bridge(self)
        from .m07_bridge import M07Bridge
        self.m07=M07Bridge(self)
        from .m06_bridge import M06Bridge
        self.m06=M06Bridge(self)

    def invoke(self,body):
        op=body.get('op')
        if op=='local_storage_status':return self.service.get('/api/v1/jobs/local-storage')
        if op=='local_storage_policy':
            return self.service.request('/api/v1/jobs/local-storage/policy','POST',body['policy'])
        if op=='local_storage_cleanup':
            return self.service.request('/api/v1/jobs/local-storage/cleanup','POST')
        if op=='egg_preview_open':
            try:
                with self.provider.audio_stream(body['id']) as stream:
                    from .annotation_audio import source_hash
                    sha=source_hash(stream)
                    value=self.service.egg_preview('open',stream)
            except ValueError as exc: raise FileAccessError(str(exc)) from None
            if value['sha256']!=sha: raise FileAccessError('音频校验失败。')
            self.egg_preview_sources={value['session_id']:body['id']}
            return value
        if op=='egg_preview_update':
            session=str(UUID(body['session']))
            file_id=self.egg_preview_sources.get(session)
            if file_id is None: raise FileAccessError('egg_preview_expired')
            # Recheck the existing grant and fingerprint without rereading WAV.
            self.provider.validate(file_id)
            try: return self.service.egg_preview('update',body['config'],session)
            except ValueError as exc: raise FileAccessError(str(exc)) from None
        if op=='egg_preview_close':
            session=str(UUID(body['session']))
            self.egg_preview_sources.pop(session,None)
            return self.service.egg_preview('close',session=session)
        if op=='research_audio':return self.provider.audio_preview(body['id'])
        if op=='research_scan':return self.provider.scan(body['id'])
        if isinstance(op,str) and op.startswith('m05_'):return self.m05.invoke(body)
        if isinstance(op,str) and op.startswith('m11_'):return self.m11.invoke(body)
        if isinstance(op,str) and op.startswith('m07_'):return self.m07.invoke(body)
        if isinstance(op,str) and op.startswith('m06_'):return self.m06.invoke(body)
        if isinstance(op,str) and op.startswith('m14_'):return self.m14.invoke(body)
        if isinstance(op,str) and op.startswith('m08_'):return self.m08.invoke(body)
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
            ref=self.stream_input(body['id'],'audio')
            return self.service.request('/api/v1/jobs/egg/create','POST',dict(project_id=PROJECT,
                idempotency_key=body['key'],audio=ref,config=body['config']))
        if op=='egg_jobs':return [j for j in self.service.get('/api/v1/jobs?project_id='+PROJECT)['jobs'] if j['operation']=='egg_analysis']
        if op in ('reconstruct','reconstruction_preview'):
            raw,_=self.provider.read(body['id']);entry=self.provider.entries[body['id']]
            ref=self.service.import_input(raw,entry.name,'audio' if body['config'].get('mode')=='audio_draw' else 'image')
            endpoint='preview' if op=='reconstruction_preview' else 'create'
            from urllib.error import HTTPError
            try:return self.service.request('/api/v1/jobs/spec2wav/'+endpoint,'POST',dict(project_id=PROJECT,idempotency_key=body.get('key') or uuid4().hex,image=ref,config=body['config']))
            except HTTPError as exc:
                code=json.load(exc).get('detail')
                message={'invalid_spectrogram_audio':'音频需为 30 秒内、8000–96000 Hz 的单声道或双声道文件。','invalid_image_corners':'四点须按左上、右上、右下、左下围成有效区域。','spectrogram_budget':'输入或笔迹超过预算，请缩小图片、缩短音频或减少笔迹。','preview_busy':'另一个预览正在处理，请稍后重试。','preview_timeout':'频谱预览超时，请缩小输入后重试。','input_unavailable':'源文件已失效，请重新选择。'}.get(code if isinstance(code,str) else '', '频谱参数或输入无效，请检查范围后重试。')
                raise FileAccessError(message) from None
        if op=='reconstructions':return [j for j in self.service.get('/api/v1/jobs?project_id='+PROJECT)['jobs'] if j['operation']=='spectrogram_to_audio']
        if op=='cancel_job':return self.service.request('/api/v1/jobs/'+str(UUID(body['id']))+'/cancel','POST')
        if op=='save_job':return self.save(str(UUID(body['id'])),body['directory'],single=True)
        if op=='result':
            import base64
            job=self.service.get('/api/v1/jobs/'+str(UUID(body['job'])))
            if job['state']!='succeeded' or job['operation'] not in ('spectrogram_to_audio','egg_analysis','lpc_analysis','pitch_manipulation','phonology_induction','speech_synthesis','phonation_synthesis','mfa_alignment','lip_analysis'):raise FileAccessError('分析结果尚不可用。')
            file=next((f for f in job['result_manifest']['files'] if f['id']==body['id']),None)
            if not file or file['size_bytes']>64_000_000:raise FileAccessError('结果文件不正确。')
            chunks=[self.service.binary(f'/api/v1/jobs/local-results/{file["id"]}?offset={offset}&size={min(1048576,file["size_bytes"]-offset)}') for offset in range(0,file['size_bytes'],1048576)]
            raw=b''.join(chunks)
            if len(raw)!=file['size_bytes'] or hashlib.sha256(raw).hexdigest()!=file['sha256']:raise FileAccessError('结果校验失败。')
            return {'base64':base64.b64encode(raw).decode(),'sha256':file['sha256']}
        if op=='parameters':
            entry=self.provider.entries.get(body['id'])
            if entry and entry.name.lower().endswith(('.xlsx','.ptb.sqlite','.ptb.sqlite3')):
                ref=self.stream_input(body['id'],'parameter_bundle')
                from urllib.error import HTTPError
                try:return self.service.request('/api/v1/jobs/local-parameter-window','POST',dict(asset_id=ref['asset_id'],view=body.get('view')))
                except HTTPError as error:
                    code=json.load(error).get('detail','')
                    message={'preview_busy':'已有预览正在读取，请稍后重试。','parameter_read_timeout':'参数表索引超时，请优先打开同名 SQLite 结果。',
                        'parameter_input_budget':'参数文件超过读取预算，请优先打开同名 SQLite 结果。','invalid_parameter_table':'参数文件结构不受支持或包含公式。',
                        'parameter_read_failed':'参数读取进程未完成，请重新打开结果。'}.get(code if isinstance(code,str) else '', '参数窗口读取失败，请检查本机任务服务。')
                    raise FileAccessError(message) from None
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
            self.provider.validate(body['id']);entry=self.provider.entries[body['id']]
            h=hashlib.sha256();directory=self.provider.directory(entry.directory)
            with open_locked(directory.path/entry.name,directory.path,entry.fingerprint) as stream:
                for raw in iter(lambda:stream.read(1048576),b''):h.update(raw)
            sha=h.hexdigest()
            return self.service.get('/api/v1/jobs/parents/latest?project_id='+PROJECT+'&sha256='+sha)
        if op in ('get','cancel','job','retry'):
            key=str(UUID(body['id']))
            path='/api/v1/jobs/'+(key if op in ('job','retry') else 'batches/'+key)
            if op in ('cancel','retry'):path+='/'+op
            return self.service.request(path,'POST' if op in ('cancel','retry') else 'GET',{'idempotency_key':body['key']} if op=='retry' else None)
        if op=='submit':
            if set(body)-{'op','operation','inputs','config','layer','idempotency_key'} or not isinstance(body.get('inputs'),list) or not 1<=len(body['inputs'])<=1000:
                raise FileAccessError('批次请求不正确。')
            inputs=[];sources=[]
            for item in body['inputs']:
                if set(item)-{'audio','textgrid','lip','parent_result','legacy_result'}:raise FileAccessError('不支持的关联。')
                mapped={}
                for role,key in item.items():
                    if key is None:continue
                    if role=='parent_result':
                        if set(key)!={'asset_id','sha256'}:raise FileAccessError('参数来源不正确。')
                        mapped[role]=key;continue
                    if role=='lip':
                        raw,name,_=self.lip_input(key)
                    elif role=='audio' and ((body.get('config') or {}).get('extended') or item.get('parent_result')):
                        mapped[role]=self.stream_input(key,'audio');continue
                    else:
                        raw,_=self.provider.read(key);name=self.provider.entries[key].name
                    mapped[role]=self.service.import_input(raw,name,role)
                inputs.append(mapped)
                sources.append(self.provider.entries[item['audio']].directory)
            batch=self.service.request('/api/v1/jobs/batches/create','POST',dict(project_id=PROJECT,operation=body['operation'],inputs=inputs,
                config=body.get('config'),layer=body.get('layer'),idempotency_key=body['idempotency_key']))
            self.batch_sources[batch['id']]=sources
            return batch
        if op=='save':return self.save(str(UUID(body['id'])),body['directory'],beside_sources=body.get('beside_sources') is True)
        raise FileAccessError('不支持的任务操作。')

    def stream_input(self,key,role):
        self.provider.validate(key);entry=self.provider.entries[key]
        cached=self.stream_sources.get((key,role))
        if cached:return cached
        directory=self.provider.directory(entry.directory)
        with open_locked(directory.path/entry.name,directory.path,entry.fingerprint) as stream:
            if role=='audio':
                import soundfile as sf
                from phonetic_core.models.audio_bounds import validate_source
                with sf.SoundFile(stream,closefd=False) as source:
                    try:validate_source(source.frames,source.samplerate,source.channels)
                    except ValueError as error:
                        raise FileAccessError(entry.name+'：'+('最长支持 30 分钟，请先切分音频。' if str(error)=='m01_duration_limit' else '音频采样率或声道格式不受支持。')) from None
                stream.seek(0)
            result=self.service.import_stream(stream,entry.name,role,entry.fingerprint[2])
        self.stream_sources[(key,role)]=result
        return result

    def lip_input(self,file_id):
        """Read JSON or convert legacy lip data without writing to its directory."""
        raw,_=self.provider.read(file_id);entry=self.provider.entries[file_id]
        if entry.name.lower().endswith('.lip.json'):return raw,entry.name,False
        if not entry.name.lower().endswith('.pkl') or entry.name.lower().endswith('_timestamps.pkl'):raise FileAccessError('请选择旧唇形 PKL。')
        name=entry.name[:-4]+'.lip.json'
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
        # Recheck both granted inputs after the bounded conversion child exits.
        if self.provider.read(file_id)[0]!=raw:raise FileAccessError('旧 PKL 已变化，请刷新。')
        if companions and self.provider.read(companions[0]['id'])[0]!=companion:raise FileAccessError('伴随时间戳已变化。')
        return converted,name,bool(companions)

    def convert_lip(self,file_id):
        entry=self.provider.entries.get(file_id)
        if not entry or not entry.name.lower().endswith('.pkl') or entry.name.lower().endswith('_timestamps.pkl'):raise FileAccessError('请选择旧唇形 PKL。')
        converted,name,companion_found=self.lip_input(file_id)
        directory=self.provider.directory(entry.directory);root=directory.path;target=root/name
        with pin_directory(root) as output:
            self.provider.directory(entry.directory)
            if output.exists(target):raise FileAccessError('已有同名 .lip.json，保持原样；可直接关联它。')
            temp=root/('.ptb-'+uuid4().hex+'.part');original=None
            try:
                with output.open(temp,'xb') as f:
                    original=identity(os.fstat(f.fileno()));f.write(converted);f.flush();os.fsync(f.fileno())
                output.publish(temp,target)
            finally:
                if original:output.unlink(temp,original)
            file=next(f for f in self.provider.list(entry.directory) if f['name']==name)
            return {'file':file,'companion_found':companion_found}

    def save(self,batch_id,directory_id,*,single=False,beside_sources=False):
        directory=self.provider.directory(directory_id)
        if directory.purpose not in ('input','output'):raise FileAccessError('请选择结果目录。')
        root=directory.path
        sources=self.batch_sources.get(batch_id) if beside_sources else None
        if beside_sources and sources is None:raise FileAccessError('此历史批次的子目录授权已失效，请取消结果与WAV同目录并选择结果目录。')
        if single:
            job=self.service.get('/api/v1/jobs/'+batch_id)
            if job['operation'] not in ('spectrogram_to_audio','egg_analysis','lpc_analysis','speech_synthesis','phonation_synthesis') or job['state']!='succeeded':raise FileAccessError('分析结果尚不可用。')
            batch={'summary':{'items':[dict(state='succeeded',job_id=batch_id,index=0)]}}
        else:batch=self.service.get('/api/v1/jobs/batches/'+batch_id)
        created=[];pending={};saved=[];exported_ids=[]
        def sha(path,output):
            value=hashlib.sha256()
            with output.open(path,'rb') as f:
                for block in iter(lambda:f.read(65536),b''):value.update(block)
            return value.hexdigest()
        with ExitStack() as stack:
            pinned={directory_id:stack.enter_context(pin_directory(root))}
            output=pinned[directory_id]
            self.provider.directory(directory_id)
            try:
                for item in batch['summary']['items']:
                    if item['state']!='succeeded':continue
                    target_id=sources[item['index']] if sources is not None else directory_id
                    target_directory=self.provider.directory(target_id)
                    if sources is not None:
                        if target_directory.purpose!='input':raise FileAccessError('源音频目录没有结果保存授权。')
                        target_directory.path.relative_to(directory.path)
                    root=target_directory.path
                    if target_id not in pinned:pinned[target_id]=stack.enter_context(pin_directory(root))
                    output=pinned[target_id]
                    job=self.service.get('/api/v1/jobs/'+item['job_id'])
                    export_names={}
                    if job['operation'] in ('egg_analysis','lpc_analysis'):
                        import base64
                        meta=next(f for f in job['result_manifest']['files'] if f['name']==('lpc.ptb.json' if job['operation']=='lpc_analysis' else 'egg.ptb.json'))
                        verified=self.invoke(dict(op='result',job=job['id'],id=meta['id']))
                        export_names=json.loads(base64.b64decode(verified['base64'])).get('export_names',{})
                    parameter_job=job['operation'] in ('acoustic_analysis','textgrid_segment')
                    export_files=[]
                    for file in job['result_manifest']['files']:
                        name=file['name']
                        if parameter_job and name.lower().endswith('.json'):continue
                        if job['operation'] in ('egg_analysis','lpc_analysis'):name=export_names.get(name,name)
                        if job['operation']=='acoustic_analysis':name=Path(batch['audio_names'][item['index']]).stem+name[len('result'):]
                        if not name or len(name)>220 or any(ord(c)<32 or c in '/\\:<>"|?*' for c in name):raise FileAccessError('输出文件名不受支持。')
                        export_files.append(dict(file,export_name=name))
                    exported_ids.extend(file['id'] for file in export_files)
                    paths=parameter_export_paths(export_files,root,output,sha) if parameter_job else {}
                    for file in export_files:
                        name=file['export_name'];path=paths[file['id']] if parameter_job else root/name
                        if output.exists(path):
                            checked_path(path)
                            if output.stat(path).st_size==file['size_bytes'] and sha(path,output)==file['sha256']:
                                saved.append(path.relative_to(directory.path).as_posix());continue
                            if parameter_job:raise FileAccessError('结果目录在保存期间发生变化，请重新保存。')
                            path=root/(item['job_id'][:8]+'-'+name)
                        if output.exists(path):
                            checked_path(path)
                            if output.stat(path).st_size==file['size_bytes'] and sha(path,output)==file['sha256']:
                                saved.append(path.relative_to(directory.path).as_posix());continue
                            path=root/(uuid4().hex[:8]+'-'+name)
                        temp=root/('.ptb-'+uuid4().hex+'.part')
                        digest=hashlib.sha256();offset=0
                        with output.open(temp,'xb') as f:
                            pending[temp]=(identity(os.fstat(f.fileno())),output)
                            while offset<file['size_bytes']:
                                self.provider.directory(directory_id)
                                self.provider.directory(target_id)
                                size=min(1_048_576,file['size_bytes']-offset)
                                raw=self.service.binary(f'/api/v1/jobs/local-results/{file["id"]}?offset={offset}&size={size}')
                                if len(raw)!=size:raise FileAccessError('结果下载不完整。')
                                f.write(raw);digest.update(raw);offset+=size
                            f.flush();os.fsync(f.fileno())
                        if digest.hexdigest()!=file['sha256']:raise FileAccessError('结果校验失败。')
                        self.provider.directory(directory_id)
                        self.provider.directory(target_id)
                        output.publish(temp,path)  # Atomic no-clobber on each supported platform.
                        original,_=pending.pop(temp)
                        created.append((path,original,file['sha256'],output));saved.append(path.relative_to(directory.path).as_posix())
                warning=self.record_export(exported_ids) if exported_ids else None
                answer={'saved':saved,'count':len(saved),'batch_id':batch_id}
                if warning:answer['retention_warning']=warning
                return answer
            except BaseException:
                for path,(original,output) in pending.items():
                    output.unlink(path,original)
                for path,original,digest,output in created:
                    if output.exists(path) and identity(output.stat(path))==original:
                        checked_path(path)
                        if sha(path,output)==digest:output.unlink(path,original)
                raise

    def record_export(self,asset_ids):
        """Receipt after byte verification; never accept a renderer assertion."""
        if not getattr(self.service,'local_files_root',None):return None
        ids=list(dict.fromkeys(asset_ids))
        try:
            for start in range(0,len(ids),3004):
                self.service.request('/api/v1/jobs/local-storage/export-receipt','POST',{'assets':ids[start:start+3004]})
        except Exception:
            # Saved external files are intact. Conservative unexported retention
            # continues if recording the cache receipt fails.
            return '文件已保存，但缓存导出登记失败。缓存暂按未导出结果的期限保留。'
        return None
