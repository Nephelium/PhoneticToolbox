"""Append-only PCM, durable project manifests and bounded range readers."""
from __future__ import annotations
import copy
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from phonetic_core.recording.edits import frames, select

SCHEMA = 'ptb-recording/1'
CHUNK_FRAMES = 131072
MAX_MANIFEST = 32*1024*1024


def uid():return uuid.uuid4().hex
def now():return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uid()+'.tmp')
    raw=json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2).encode('utf8')
    if len(raw)>MAX_MANIFEST:raise ValueError('工程清单超过 32 MiB 安全预算')
    with tmp.open('xb') as handle:
        handle.write(raw);handle.flush();os.fsync(handle.fileno())
    os.replace(tmp,path)


def read_json(path):
    path=Path(path)
    if path.stat().st_size>MAX_MANIFEST:raise ValueError('清单过大')
    return json.loads(path.read_text(encoding='utf8'))


def safe_path(root, relative):
    root=Path(root).resolve()
    if not isinstance(relative,str) or '\\' in relative or ':' in relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('工程资源路径无效')
    path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('工程资源越界')
    return path


def safe_name(value):
    text=re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',str(value)).strip(' .')[:100]
    if not text or re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)',text,re.I):text='recording_'+text
    return text


def save_pcm(root, relative, array):
    x=np.asarray(array,dtype='<f4',order='C')
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('无效原始采样块')
    path=safe_path(root,relative);path.parent.mkdir(parents=True,exist_ok=True)
    raw=x.tobytes();digest=hashlib.sha256(raw).hexdigest()
    # Exclusive create protects earlier raw even after a retry or ID collision.
    with path.open('xb') as handle:
        count=handle.write(raw)
        if count!=len(raw):raise OSError('音频短写，保留恢复数据')
        handle.flush();os.fsync(handle.fileno())
    if path.stat().st_size!=len(raw):raise OSError('音频落盘大小错误')
    return {'file':relative,'start':0,'end':len(x),'frames':len(x),'channels':x.shape[1],'sha256':digest}


def validate_span(root, span, *, hash_check=False):
    path=safe_path(root,span['file'])
    n,c=span['frames'],span['channels']
    if not isinstance(n,int) or n<1 or not isinstance(c,int) or not 1<=c<=8 or not 0<=span['start']<=span['end']<=n:
        raise ValueError('工程片段帧索引无效')
    if path.stat().st_size!=n*c*4:raise ValueError('音频分块大小与清单不一致')
    if hash_check:
        digest=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=span['sha256']:raise ValueError('音频分块校验失败')


def iter_audio(root, spans, block_frames=CHUNK_FRAMES):
    for span in spans:
        validate_span(root,span)
        c=span['channels'];remaining=span['end']-span['start']
        with safe_path(root,span['file']).open('rb') as handle:
            handle.seek(span['start']*c*4)
            while remaining:
                count=min(remaining,block_frames);raw=handle.read(count*c*4)
                if len(raw)!=count*c*4:raise ValueError('音频分块读取不完整')
                x=np.frombuffer(raw,dtype='<f4').reshape(count,c).copy()
                if not np.isfinite(x).all():raise ValueError('音频含非有限值')
                yield x;remaining-=count


def read_range(root,spans,start,end,limit=1_000_000):
    if end-start>limit:raise ValueError('请求超过有界音频窗口')
    selected=select(spans,start,end)
    data=list(iter_audio(root,selected))
    return np.concatenate(data) if data else np.empty((0,spans[0]['channels'] if spans else 1),np.float32)


class Project:
    def __init__(self,root,create=False):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        if create and any(path.name!='.m16-writer.lock' for path in self.root.iterdir()):
            raise ValueError('新工程需要空目录，请选择一个新的空文件夹')
        self.lock_file=None
        # OS-level lock releases on crash. File remains to avoid unlink-lock races.
        lock=self.root/'.m16-writer.lock'
        self.lock_file=lock.open('a+b');self.lock_file.seek(0)
        if lock.stat().st_size==0:self.lock_file.write(b'0');self.lock_file.flush()
        try:
            if os.name=='nt':
                import msvcrt
                self.lock_file.seek(0);msvcrt.locking(self.lock_file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.lock_file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.lock_file.close();self.lock_file=None;raise ValueError('工程已在另一个窗口打开，当前拒绝第二写者')
        try:
            path=self.root/'project.json'
            if path.exists():
                if create:raise ValueError('该目录已有录音工程，请使用打开工程')
                self.data=read_json(path);self.validate()
            elif create:
                self.data={'schema_version':SCHEMA,'id':uid(),'created_at':now(),'revision':0,'tasks':[],'takes':[],'selected':{},'recoveries':[]}
                self.commit(self.data)
            else:raise ValueError('目录中没有 recording/1 工程')
            self.scan_recovery()
        except BaseException:self.close();raise

    def validate(self):
        d=self.data
        if d.get('schema_version')!=SCHEMA or not isinstance(d.get('takes'),list) or not isinstance(d.get('tasks'),list):raise ValueError('工程版本或结构不支持')
        if len(d['takes'])>10000:raise ValueError('工程 take 数量超过 10000')
        seen=set()
        for take in d['takes']:
            if take['id'] in seen:raise ValueError('工程含重复 take ID')
            seen.add(take['id'])
            for version in take['versions']:
                for span in version['spans']:validate_span(self.root,span)
            if not 0<=take['head']<len(take['versions']):raise ValueError('工程版本索引无效')

    def commit(self,data):
        candidate=copy.deepcopy(data);candidate['revision']=self.data.get('revision',0)+1;candidate['updated_at']=now()
        atomic_json(self.root/'manifests'/f"{candidate['revision']:08d}-{uid()}.json",candidate)
        atomic_json(self.root/'project.json',candidate)
        self.data=candidate

    def scan_recovery(self):
        known={t['id'] for t in self.data['takes']};recovered=[]
        for path in (self.root/'recovery').glob('*/capture.json'):
            try:
                item=read_json(path)
                if item['id'] in known:continue
                valid=[]
                for span in item.get('spans',[]):
                    try:validate_span(self.root,span,hash_check=True);valid.append(span)
                    except (OSError,ValueError):break
                if valid:recovered.append({**item,'spans':valid,'status':'recovered_partial','error':'异常中断恢复，仅包含已落盘且哈希通过的前缀，末尾可能缺失'})
            except (OSError,ValueError,KeyError):continue
        self.recoveries=recovered

    def close(self):
        if self.lock_file:
            try:
                if os.name=='nt':
                    import msvcrt
                    self.lock_file.seek(0);msvcrt.locking(self.lock_file.fileno(),msvcrt.LK_UNLCK,1)
                else:
                    import fcntl
                    fcntl.flock(self.lock_file.fileno(),fcntl.LOCK_UN)
            finally:self.lock_file.close();self.lock_file=None
