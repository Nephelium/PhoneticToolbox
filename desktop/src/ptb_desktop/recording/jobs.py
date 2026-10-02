"""Owned local processing subprocess and streaming export worker functions."""
from __future__ import annotations
import copy
import csv
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from phonetic_core.recording import noise_profile, denoise, gain_audio
from phonetic_core.recording.edits import frames, select
from .storage import read_range, iter_audio, save_pcm, atomic_json, safe_name, uid, validate_span


def process_audio(root,request,result_path,cancel_path):
    """Only this subprocess computes denoise; parent publishes after verification."""
    try:
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x00004000)
        else:
            try:os.nice(5)
            except OSError:pass
        root=Path(root);source=request['spans'];total=frames(source);cfg=request['config'];channels=cfg['channels']
        eligible=[i for i,r in enumerate(cfg['roles']) if r=='microphone'];mic=request.get('channels',eligible)
        if not isinstance(mic,list) or len(set(mic))!=len(mic) or any(i not in eligible for i in mic):raise ValueError('处理通道必须是用户选择的音频角色，EGG 不允许处理')
        if not mic:raise ValueError('当前录音没有麦克风通道；EGG 通道禁止降噪和数字增益处理')
        profile=None;meta={}
        if request['kind']=='denoise':
            noise=request['noise'];raw=read_range(root,noise['spans'],noise['start'],noise['end'],limit=cfg['sample_rate']*60)
            profile,meta=noise_profile(raw[:,noise['channel']])
            meta.update({'strength':request['strength'],'noise':noise,'processed_channels':mic,'boundary':'1024-frame halo; 131072-frame core blocks; zero pad at outer endpoints'})
        else:meta={'algorithm':'linear-gain/1','gain_db':request['gain_db'],'processed_channels':mic}
        spans=[];block_frames=131072;halo=1024
        for start in range(0,total,block_frames):
            if Path(cancel_path).exists():raise InterruptedError('处理已取消，原始和当前版本保留')
            end=min(total,start+block_frames);left=max(0,start-halo);right=min(total,end+halo)
            x=read_range(root,source,left,right,limit=block_frames+2*halo)
            y=denoise(x,profile,mic,request['strength']) if profile is not None else gain_audio(x,request['gain_db'],mic)
            y=y[start-left:end-left]
            original=x[start-left:end-left]
            untouched=[i for i in range(channels) if i not in mic]
            if y.shape!=original.shape or not np.isfinite(y).all() or (untouched and not np.array_equal(y[:,untouched],original[:,untouched])):
                raise ValueError('处理输出帧长或 EGG 不变量验证失败')
            span=save_pcm(root,f"derived/{request['id']}/{len(spans):06d}.f32",y)
            validate_span(root,span,hash_check=True);spans.append(span)
            atomic_json(result_path,{'state':'running','progress':end/max(1,total)})
        atomic_json(result_path,{'state':'complete','spans':spans,'metadata':meta,'frames':total})
    except BaseException as exc:
        atomic_json(result_path,{'state':'cancelled' if isinstance(exc,InterruptedError) else 'failed','error':str(exc)})


def export_audio(root,items,target,subtype,result_path,cancel_path,*,max_frames=16_000_000,task_summary=None):
    """At most 256 MiB per 4-channel float WAV; long outputs remain segmented."""
    output=[];target=Path(target)
    batch=target/('录音导出-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uid()[:6])
    try:
        target.mkdir(parents=True,exist_ok=True);batch.mkdir()
        if subtype not in ('FLOAT','PCM_24','PCM_16'):raise ValueError('不支持的 WAV 格式')
        for index,item in enumerate(items):
            if Path(cancel_path).exists():break
            entry={'take_id':item['id'],'task_snapshot':item.get('task_snapshot'),'version_id':item['version']['id'],'status':'failed','files':[],
                   'sample_rate':item['config']['sample_rate'],'roles':item['config']['roles'],'subtype':subtype,'quality':item.get('quality'),
                   'processing':item['version'].get('metadata',{}),'frame_count':frames(item['version']['spans'])}
            try:
                cfg=item['config'];spans=item['version']['spans'];total=frames(spans)
                # PCM is never silently clipped; caller can explicitly apply a gain version.
                if subtype!='FLOAT':
                    for block in iter_audio(root,spans):
                        positive_limit=1-2**(-15 if subtype=='PCM_16' else -23)
                        if np.max(block,initial=0)>positive_limit or np.min(block,initial=0)<-1:raise ValueError('PCM 导出超过目标量化格式范围，请选择 float32 或先显式减小数字增益')
                chunk_limit=min(max_frames,256*1024*1024//(cfg['channels']*4))
                for part,start in enumerate(range(0,total,chunk_limit)):
                    if Path(cancel_path).exists():raise InterruptedError('批量导出已取消，已完成文件保留')
                    end=min(total,start+chunk_limit);name=f"{index+1:04d}_{safe_name(item.get('name') or 'recording')}_{item['id'][:6]}_{item['version']['kind']}"
                    if total>chunk_limit:name+=f'_part{part+1:04d}'
                    path=batch/(name+'.wav')
                    partial=path.with_name(path.name+'.partial')
                    entry.setdefault('partial_files',[]).append(partial.name)
                    # Exclusive open plus an explicit file descriptor prevents overwrite races.
                    with partial.open('xb') as file_handle:
                        with sf.SoundFile(file_handle,mode='w',samplerate=cfg['sample_rate'],channels=cfg['channels'],format='WAV',subtype=subtype,closefd=False) as handle:
                            for block in iter_audio(root,select(spans,start,end)):
                                if Path(cancel_path).exists():raise InterruptedError('导出已取消')
                                handle.write(block)
                        file_handle.flush();os.fsync(file_handle.fileno())
                    info=sf.info(partial)
                    if info.frames!=end-start or info.samplerate!=cfg['sample_rate'] or info.channels!=cfg['channels'] or info.subtype!=subtype:raise ValueError('导出文件回读格式验证失败')
                    digest=hashlib.sha256()
                    with partial.open('rb') as handle:
                        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
                    if not partial.resolve().is_relative_to(batch.resolve()) or not path.resolve().is_relative_to(batch.resolve()) or path.exists():raise ValueError('导出发布路径无效或已占用')
                    partial.rename(path)
                    entry['partial_files'].remove(partial.name)
                    entry['files'].append({'name':path.name,'start_frame':start,'end_frame':end,'sha256':digest.hexdigest()})
                entry['status']='exported'
            except BaseException as exc:entry['error']=str(exc)
            output.append(entry)
            atomic_json(result_path,{'state':'running','progress':(index+1)/max(1,len(items)),'items':output,'directory_name':batch.name})
        for item in items[len(output):]:output.append({'take_id':item['id'],'status':'cancelled','files':[]})
        manifest={'schema_version':'ptb-recording-export/1','items':output,'task_summary':task_summary or []}
        atomic_json(batch/'manifest.json',manifest)
        with (batch/'manifest.csv').open('x',encoding='utf-8-sig',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['take_id','task_id','prompt','version','status','file','start_frame','end_frame','sha256','error'])
            for item in output:
                task=item.get('task_snapshot') or {}
                def safe(value):
                    text=str(value or '')
                    return "'"+text if text.startswith(('=','+','-','@','\t','\r')) else text
                for file in item['files'] or [{}]:writer.writerow([item['take_id'],safe(task.get('id')),safe(task.get('prompt')),item.get('version_id',''),item['status'],file.get('name',''),file.get('start_frame',''),file.get('end_frame',''),file.get('sha256',''),safe(item.get('error'))])
            for task in task_summary or []:
                if task['status']!='selected':writer.writerow(['',safe(task['id']),safe(task.get('prompt')),'',task['status'],'','','','',''])
        state='cancelled' if Path(cancel_path).exists() else 'complete'
        atomic_json(result_path,{'state':state,'items':output,'directory_name':batch.name,'success_count':sum(x['status']=='exported' for x in output),'total':len(items)})
    except BaseException as exc:atomic_json(result_path,{'state':'failed','items':output,'error':str(exc),'directory_name':batch.name})
