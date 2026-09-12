"""One native process per owning window, with bounded startup and shutdown."""
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import uuid
from .process_guard import OwnedJob


class VocalTractClient:
    def __init__(self,resources,profile,*,legacy_profile=None,playback_allowed=True):
        self.config={'resources':str(resources),'profile':str(profile),'legacy_profile':str(legacy_profile) if legacy_profile else None,
            'parent_pid':os.getpid(),'playback_allowed':playback_allowed}
        self.process=None;self.job=None;self.start_lock=threading.RLock();self.write_lock=threading.Lock();self.pending={};self.pending_lock=threading.Lock()

    def start(self):
        with self.start_lock:
            if self.process and self.process.poll() is None:return
            command=([sys.executable,'--m10-worker'] if getattr(sys,'frozen',False) else [sys.executable,'-X','utf8','-m','ptb_desktop.vocal_tract.worker'])
            profile=Path(self.config['profile']);profile.mkdir(parents=True,exist_ok=True)
            with (profile/'native-worker.log').open('w',encoding='utf-8') as log:
                p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.process=p
            try:
                self.job=OwnedJob(p);ready=queue.Queue()
                def read():
                    try:
                        for line in p.stdout:
                            try:data=json.loads(line)
                            except ValueError:continue
                            if 'ready' in data:ready.put(data);continue
                            with self.pending_lock:item=self.pending.pop(data.get('id'),None)
                            if item:item[1].put(data)
                    except (UnicodeError,OSError):
                        logging.getLogger(__name__).warning('M10 native channel closed or invalid UTF-8')
                    finally:
                        if p.poll() is None:p.terminate()
                        ready.put({'error':'声道引擎启动失败，请检查本机日志。'})
                        with self.pending_lock:
                            for key,(owner,q) in list(self.pending.items()):
                                if owner is p:
                                    q.put({'ok':False,'error':'声道引擎已退出，请重新打开模块。'});self.pending.pop(key)
                threading.Thread(target=read,daemon=True).start()
                p.stdin.write(json.dumps(self.config)+'\n');p.stdin.flush()
                if ready.get(timeout=30).get('ready')!='m10/1':raise RuntimeError('声道引擎启动失败，请检查 native-worker.log。')
            except Exception:self.close();raise

    def invoke(self,op,body=None):
        if op=='shutdown':self.close();return {'closed':True}
        request_id=uuid.uuid4().hex;q=queue.Queue()
        data=json.dumps({'id':request_id,'op':op,'body':body or {}},ensure_ascii=False,allow_nan=False)
        if len(data)>8_000_000:raise ValueError('姿势序列超过消息容量，请分段保存。')
        try:
            with self.start_lock:
                self.start();p=self.process
                with self.pending_lock:self.pending[request_id]=(p,q)
                with self.write_lock:p.stdin.write(data+'\n');p.stdin.flush()
            result=q.get(timeout=150)
            if not result.get('ok'):raise RuntimeError(result.get('error','声道请求失败'))
            return result['value']
        finally:
            with self.pending_lock:self.pending.pop(request_id,None)

    def close(self):
        with self.start_lock:self._close()

    def _close(self):
        p=self.process;self.process=None
        if p:
            try:p.stdin.close()
            except OSError:pass
            try:p.wait(timeout=4)
            except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=4)
            p.stdout.close()
        if self.job:self.job.close();self.job=None
