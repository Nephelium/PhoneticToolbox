"""Local M16 project owner. All references resolve inside user-authorized roots."""
from __future__ import annotations
import copy
import multiprocessing
import shutil
import threading
import time
import sys
from pathlib import Path
import numpy as np
from phonetic_core.recording import meter
from phonetic_core.recording.signal import DisplaySpectrum
from phonetic_core.recording.edits import frames,select,remove,insert
from .storage import Project,uid,now,read_range,iter_audio,atomic_json,read_json,safe_name
from .capture import Capture
from .devices import devices,validate_config,Playback,PlaybackStartError
from .jobs import process_audio,export_audio


class RecordingService:
    def __init__(self,*,backend=None):
        self.backend=backend;self.project=None;self.capture=None;self.player=None;self.job=None;self.clipboard=None;self.grants={};self.noise=None
        self.lock=threading.RLock()

    @property
    def recording(self):return bool(self.capture and not self.capture.probe)

    def grant(self,path,purpose):
        token=uid();self.grants[token]=(Path(path).resolve(),purpose);return {'grant':token,'label':Path(path).name}

    def granted(self,token,purposes):
        value=self.grants.get(token)
        if value is None or value[1] not in purposes:raise ValueError('目录授权无效，请重新选择')
        return value[0]

    def require(self,idle=False):
        if self.project is None:raise ValueError('请先新建或打开本地录音工程')
        if idle and self.capture:raise ValueError('请先停止录音或录前检测')
        if idle and self.job:raise ValueError('后台处理或导出尚未结束，请等待或取消')
        return self.project

    def take(self,take_id):
        p=self.require()
        item=next((t for t in p.data['takes'] if t['id']==take_id),None)
        if item is None:raise ValueError('录音 take 不存在')
        return item

    def view(self):
        if not self.project:return None
        data=copy.deepcopy(self.project.data)
        for take in data['takes']:
            for version in take['versions']:
                version['frames']=frames(version['spans']);version.pop('spans',None)
        data['label']=self.project.root.name;data['recoveries']=[{'id':r['id'],'frames':frames(r['spans']),'error':r['error']} for r in self.project.recoveries]
        return data

    def append_version(self,take_id,spans,kind,metadata=None):
        p=self.require();data=copy.deepcopy(p.data);take=next(t for t in data['takes'] if t['id']==take_id)
        if frames(spans)<=0:raise ValueError('操作会产生空录音，已保留当前版本')
        version={'id':uid(),'kind':kind,'created_at':now(),'spans':copy.deepcopy(spans),'metadata':metadata or {},'parent':take['versions'][take['head']]['id']}
        # All abandoned redo heads remain in history; navigation is a separate stack.
        take['versions'].append(version);take['undo']=take.get('undo',[])+[take['head']];take['redo']=[];take['head']=len(take['versions'])-1
        p.commit(data);return self.view()

    def dispatch(self,body):
        with self.lock:return self._dispatch(body)

    def _dispatch(self,b):
        op=b.get('op')
        if op=='capabilities':
            if sys.platform!='win32' and self.backend is None:return {'available':False,'schema':'ptb-recording/1','devices':[],'device_error':'当前平台尚未通过原生录音准入，Windows 桌面为首版开发目标','project':self.view(),'local_only':True}
            try:found=devices(self.backend);reason=''
            except Exception as exc:found=[];reason='原生设备库不可用：'+str(exc)
            return {'available':True,'schema':'ptb-recording/1','devices':found,'device_error':reason,'project':self.view(),'local_only':True}
        if op=='devices':return devices(self.backend)
        if op=='open':
            if self.capture or self.job:raise ValueError('请先停止采集和后台处理')
            self.stop_play();root=self.granted(b['grant'],('new','open'))
            if self.project and self.project.root==root:return self.view()
            new=Project(root,bool(b.get('create')))
            if self.project:self.project.close()
            self.project=new;self.noise=None;self.clipboard=None;return self.view()
        if op=='project':return self.view()
        if op=='save':
            p=self.require();p.commit(p.data);return self.view()
        if op=='tasks':
            p=self.require(idle=True);tasks=b['tasks']
            if not isinstance(tasks,list) or len(tasks)>10000:raise ValueError('任务清单超过 10000 条')
            seen=set();stems=set()
            for task in tasks:
                if not isinstance(task,dict) or not isinstance(task.get('id'),str) or not task['id'] or task['id'] in seen:raise ValueError('任务编号为空或重复')
                seen.add(task['id'])
                for field in ('prompt','title','filename_stem','group','note'):
                    if not isinstance(task.get(field,''),str) or len(task.get(field,''))>20000:raise ValueError('任务文本无效或过长')
                stem=task.get('filename_stem','')
                if stem and (safe_name(stem)!=stem or stem.casefold() in stems):raise ValueError('文件名不安全或重复')
                if stem:stems.add(stem.casefold())
                if not isinstance(task.get('enabled',True),bool) or not isinstance(task.get('skipped',False),bool):raise ValueError('任务状态无效')
            data=copy.deepcopy(p.data);data['tasks']=copy.deepcopy(tasks);p.commit(data);return self.view()
        if op in ('start','probe_start'):
            if sys.platform!='win32' and self.backend is None:raise ValueError('当前平台尚未开放原生录音能力')
            p=self.require(idle=True);self.stop_play();config=validate_config(b['config'],self.backend)
            task=next((t for t in p.data['tasks'] if t['id']==b.get('task_id')),None)
            if task and not task.get('enabled',True):raise ValueError('当前任务未启用')
            capture=Capture(p.root,config,task,probe=op=='probe_start',backend=self.backend)
            self.capture=capture
            try:capture.start()
            except Exception:
                if not capture.spans and capture.stream is None and not (capture.thread and capture.thread.is_alive()):self.capture=None
                raise
            return {'id':capture.id,'config':config}
        if op in ('stop','probe_stop'):return self.stop_capture()
        if op=='status':
            status={'recording':False,'probing':False,'project_revision':self.project.data['revision'] if self.project else None,'playback':{'playing':bool(self.player and not self.player.finished.is_set()),'frame':self.player.position if self.player else 0,'error':self.player.error if self.player else ''},'job':self.poll_job()}
            if self.capture:
                channel=int(b.get('channel',0))
                status.update(self.capture.preview(bool(b.get('spectrum')),channel,b.get('gain_preview_db')))
                status['capture_active']=True
            if self.project:
                free=shutil.disk_usage(self.project.root).free;status['disk_free']=free
            return status
        if op=='recover':
            p=self.require(idle=True);item=next((r for r in p.recoveries if r['id']==b['id']),None)
            if not item:raise ValueError('恢复条目不存在')
            self.commit_capture(item);p.scan_recovery();return self.view()
        if op=='select_take':
            p=self.require(idle=True);take=self.take(b['id']);data=copy.deepcopy(p.data);key=(take.get('task_snapshot') or {}).get('id','__free__');data['selected'][key]=take['id'];p.commit(data);return self.view()
        if op=='preview':
            take=self.take(b['id']);v=take['versions'][int(b.get('version',take['head']))];n=frames(v['spans']);start=max(0,int(b.get('start',0)));end=min(n,int(b.get('end',n)));width=min(1600,max(32,int(b.get('width',900))))
            if not 0<=start<end<=n:raise ValueError('预览范围为空或越界')
            # Stream decimation for arbitrarily long takes. No whole-take RAM read.
            channels=take['config']['channels'];bins=min(width,end-start);edges=np.linspace(start,end,bins+1,dtype=np.int64);lo=np.full((bins,channels),np.inf);hi=np.full((bins,channels),-np.inf);offset=start
            display=DisplaySpectrum(end-start,take['config']['sample_rate'],int(b.get('channel',0)),width) if b.get('spectrum') else None
            for block in iter_audio(self.project.root,select(v['spans'],start,end)):
                if display is not None:display.add(block)
                stop=offset+len(block);first=max(0,int(np.searchsorted(edges,offset,side='right')-1));last=min(bins,int(np.searchsorted(edges,stop,side='left'))+1)
                for index in range(first,last):
                    a=max(int(edges[index]),offset)-offset;c=min(int(edges[index+1]),stop)-offset
                    if c>a:lo[index]=np.minimum(lo[index],block[a:c].min(axis=0));hi[index]=np.maximum(hi[index],block[a:c].max(axis=0))
                offset=stop
            out={'wave':[np.stack((lo[:,c],hi[:,c]),axis=1).tolist() for c in range(channels)],'window_frames':end-start,'frames':n,'start_frame':start,'sample_rate':take['config']['sample_rate'],'spectrum':None}
            if display is not None:
                out['spectrum']=display.result();out['spectrum_window_frames']=end-start
            return out
        if op in ('edit','undo','redo','restore','version'):
            self.require(idle=True);self.stop_play();take=self.take(b['id']);spans=take['versions'][take['head']]['spans']
            if op=='restore':return self.append_version(take['id'],take['versions'][0]['spans'],'restored_raw')
            if op in ('undo','redo','version'):
                data=copy.deepcopy(self.project.data);target=next(t for t in data['takes'] if t['id']==take['id']);key='undo' if op=='undo' else 'redo';other='redo' if op=='undo' else 'undo'
                if op=='version':
                    index=int(b['index'])
                    if not 0<=index<len(target['versions']):raise ValueError('历史版本不存在')
                    target['undo']=target.get('undo',[])+[target['head']];target['redo']=[];target['head']=index
                else:
                    if not target.get(key):return self.view()
                    target.setdefault(other,[]).append(target['head']);target['head']=target[key].pop()
                self.project.commit(data);return self.view()
            start,end=int(b['start']),int(b['end']);part=select(spans,start,end);action=b['action']
            if action in ('copy','cut'):
                if not part:return self.view()
                self.clipboard={'spans':copy.deepcopy(part),'config':copy.deepcopy(take['config'])}
                if action=='copy':return self.view()
            if action=='paste':
                if not self.clipboard:raise ValueError('模块音频剪贴板为空')
                if any(take['config'][key]!=self.clipboard['config'][key] for key in ('sample_rate','channels','roles')):raise ValueError('仅允许同采样率、同通道结构和角色粘贴，不自动重采样')
                changed=insert(spans,start,self.clipboard['spans'])
            elif action=='keep':changed=part
            elif action in ('delete','cut'):
                if start==end:return self.view()
                changed=remove(spans,start,end)
            else:raise ValueError('未知编辑命令')
            return self.append_version(take['id'],changed,'edited',{'operation':action,'range':[start,end]})
        if op=='noise':
            self.require(idle=True);take=self.take(b['id']);channel=int(b['channel']);cfg=take['config'];start,end=int(b['start']),int(b['end'])
            if not 0<=channel<cfg['channels'] or cfg['roles'][channel]!='microphone':raise ValueError('噪声样本必须来自麦克风通道，EGG 禁止降噪')
            spans=take['versions'][take['head']]['spans'];select(spans,start,end)
            if not 3072<=end-start<=cfg['sample_rate']*60:raise ValueError('噪声样本需至少 3072 帧、最多 60 秒')
            from phonetic_core.recording import noise_profile
            _,meta=noise_profile(read_range(self.project.root,spans,start,end,limit=cfg['sample_rate']*60)[:,channel])
            self.noise={'take_id':take['id'],'version_id':take['versions'][take['head']]['id'],'spans':copy.deepcopy(spans),'start':start,'end':end,'channel':channel,'sample_rate':cfg['sample_rate']}
            return {k:v for k,v in self.noise.items() if k!='spans'}|{'warning':meta['warning']}
        if op=='process_start':
            p=self.require(idle=True);self.stop_play();take=self.take(b['id']);kind=b['kind']
            if kind not in ('denoise','gain'):raise ValueError('不支持的处理')
            if kind=='denoise' and (not self.noise or self.noise['sample_rate']!=take['config']['sample_rate']):raise ValueError('请先设置相同采样率的噪声样本')
            eligible=[i for i,r in enumerate(take['config']['roles']) if r=='microphone'];channels=b.get('channels',eligible)
            if not isinstance(channels,list) or not channels or any(i not in eligible for i in channels) or len(set(channels))!=len(channels):raise ValueError('请选择音频通道，EGG 通道不参与处理')
            request={'id':uid(),'kind':kind,'spans':take['versions'][take['head']]['spans'],'config':take['config'],'channels':channels,'noise':self.noise,'strength':float(b.get('strength',1)),'gain_db':float(b.get('gain_db',0))}
            folder=p.root/'recovery'/request['id'];folder.mkdir(parents=True);result=folder/'process.json';cancel=folder/'cancel'
            proc=multiprocessing.get_context('spawn').Process(target=process_audio,args=(str(p.root),request,str(result),str(cancel)),daemon=True)
            proc.start();self.job={'kind':kind,'owner':proc,'result':result,'cancel':cancel,'take_id':take['id'],'source_version':take['versions'][take['head']]['id'],'started':time.monotonic(),'cancelled':False};return {'state':'running'}
        if op=='job_cancel':
            if self.job:self.job['cancel'].touch(exist_ok=True);self.job['cancelled']=True
            return {'state':'cancelling'}
        if op=='play':
            self.require(idle=True);self.stop_play();take=self.take(b['id']);version=take['versions'][int(b.get('version',take['head']))];spans=take['versions'][0]['spans'] if b.get('variant')=='raw' else version['spans'];total=frames(spans);start,end=int(b.get('start',0)),min(total,int(b.get('end',total)));reference=None
            if start==end:start,end=0,total
            if b.get('variant')=='residual':
                parent=next((v for v in take['versions'] if v['id']==version.get('parent')),None)
                if version['kind'] not in ('denoised','gain') or not parent or frames(parent['spans'])!=total:raise ValueError('差分预听仅支持帧长一致的降噪或增益版本及其直接来源')
                reference=select(parent['spans'],start,end)
            try:self.player=Playback(self.project.root,select(spans,start,end),take['config'],b['device'],int(b.get('channel',0)),float(b.get('volume',.7)),self.backend,reference_spans=reference)
            except PlaybackStartError as exc:self.player=exc.player;raise
            return {'playing':True}
        if op=='play_stop':self.stop_play();return {'playing':False}
        if op=='export_start':return self.start_export(b)
        raise ValueError('M16 不支持该操作')

    def commit_capture(self,item):
        p=self.require();data=copy.deepcopy(p.data)
        if any(t['id']==item['id'] for t in data['takes']):return
        spans=item.pop('spans');version={'id':uid(),'kind':'raw','created_at':now(),'spans':spans,'metadata':{'capture_status':item['status']}}
        take={**copy.deepcopy(item),'versions':[version],'head':0,'undo':[],'redo':[]};data['takes'].append(take)
        task=take.get('task_snapshot') or {};data['selected'][task.get('id','__free__')]=take['id'];p.commit(data)

    def stop_capture(self):
        if not self.capture:return {'project':self.view(),'error':''}
        capture=self.capture;item=capture.stop()
        if not capture.probe:
            if item['spans']:self.commit_capture(copy.deepcopy(item))
            elif not item['error']:item['error']='未采集到有效音频，未创建空 take'
        self.capture=None;return {'project':self.view(),'error':item['error'],'frames':item['frames'],'probe':capture.probe,'quality':item['quality']}

    def stop_play(self):
        if self.player:self.player.stop();self.player=None

    def start_export(self,b):
        p=self.require(idle=True);self.stop_play();target=self.granted(b['grant'],('export',))
        if target==p.root or target.is_relative_to(p.root):raise ValueError('请将导出保存到工程之外的目录，保护原始数据')
        mode=b.get('mode','current');items=[]
        takes=p.data['takes'] if mode in ('all','versions') else [self.take(b['id'])] if mode in ('current','selection') else [t for t in p.data['takes'] if t['id'] in p.data['selected'].values() and not next((task.get('skipped',False) or not task.get('enabled',True) for task in p.data['tasks'] if task['id']==(t.get('task_snapshot') or {}).get('id')),False)]
        if mode=='selected':
            chosen={p.data['selected'].get(t['id']) for t in p.data['tasks'] if t.get('enabled',True) and not t.get('skipped')}
            takes=[t for t in takes if t['id'] in chosen]
        for take in takes:
            if mode=='versions':
                task=take.get('task_snapshot') or {}
                for version in take['versions']:
                    if frames(version['spans']):items.append({**copy.deepcopy(take),'version':copy.deepcopy(version),'name':task.get('filename_stem') or task.get('title') or '自由录音'})
                continue
            version=copy.deepcopy(take['versions'][0 if b.get('raw') else take['head']])
            if mode=='selection':version['spans']=select(version['spans'],int(b['start']),int(b['end']))
            if not frames(version['spans']):continue
            task=take.get('task_snapshot') or {};items.append({**copy.deepcopy(take),'version':version,'name':task.get('filename_stem') or task.get('title') or '自由录音'})
        if not items:raise ValueError('没有可导出的录音或选区')
        folder=p.root/'recovery'/uid();folder.mkdir(parents=True);result=folder/'export.json';cancel=folder/'cancel'
        summary=[{'id':t['id'],'prompt':t.get('prompt',''),'status':'disabled' if not t.get('enabled',True) else 'skipped' if t.get('skipped') else 'selected' if t['id'] in p.data['selected'] else 'missing'} for t in p.data['tasks']]
        worker=threading.Thread(target=export_audio,args=(p.root,items,target,b.get('subtype','FLOAT'),result,cancel),kwargs={'task_summary':summary},daemon=True,name='ptb-m16-export');worker.start()
        self.job={'kind':'export','owner':worker,'result':result,'cancel':cancel,'started':time.monotonic(),'cancelled':False};return {'state':'running','total':len(items)}

    def poll_job(self):
        if not self.job:return None
        job=self.job;alive=job['owner'].is_alive();timeout_error=None
        if time.monotonic()-job['started']>1800 and alive:
            job['cancel'].touch(exist_ok=True);job['cancelled']=True
            if hasattr(job['owner'],'terminate'):job['owner'].terminate();job['owner'].join(2)
            alive=job['owner'].is_alive()
            timeout_error='本地处理超过 30 分钟，已请求取消，原始版本保留'
        # Read after observing owner exit: reading first can retain stale running
        # progress while the worker publishes its terminal result and exits.
        value=read_json(job['result']) if job['result'].exists() else {'state':'running'}
        if timeout_error:value={'state':'cancelling' if alive else 'failed','error':timeout_error}
        if alive:
            # A written result does not mean process/thread teardown or the parent
            # project commit has completed. Keep UI and native busy gates aligned.
            value={**value,'state':'cancelling' if job['cancelled'] or value['state'] in ('cancelled','cancelling') else 'running'}
        if not alive:
            if value['state']=='running':value={'state':'failed','error':'后台处理异常退出，未提交新版本'}
            if value['state']=='complete' and job['kind']!='export':
                take=self.take(job['take_id'])
                if job['cancelled']:value={'state':'cancelled','error':'处理已取消，原始和当前版本保留'}
                elif take['versions'][take['head']]['id']!=job['source_version']:value={'state':'failed','error':'处理期间源版本变化，未提交派生结果'}
                else:
                    expected=frames(take['versions'][take['head']]['spans'])
                    if value['frames']!=expected:raise ValueError('派生处理帧数不一致')
                    self.append_version(take['id'],value['spans'],'denoised' if job['kind']=='denoise' else 'gain',value['metadata']);value={'state':'complete','project':self.view()}
            self.job=None
        return {k:v for k,v in value.items() if k not in ('spans','metadata')}

    def close(self):
        with self.lock:
            try:
                self.stop_play()
                if self.capture:
                    result=self.stop_capture()
                    # Partial raw is durable and visible; close only fails if finalization failed.
                if self.job:
                    self.job['cancel'].touch(exist_ok=True);self.job['cancelled']=True;owner=self.job['owner'];owner.join(3)
                    if owner.is_alive():
                        if hasattr(owner,'terminate'):owner.terminate();owner.join(2)
                        else:return False
                    self.poll_job()
                if self.project:self.project.close();self.project=None
                return True
            except Exception:return False
