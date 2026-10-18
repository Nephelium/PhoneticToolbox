"""Native selection and atomic files for M10, owned by one Qt window."""
import json
import threading
import uuid
from pathlib import Path
from .video import VideoSession


class VocalFiles:
    def __init__(self):self.video=None;self.lock=threading.RLock()

    def invoke(self,op,body,client,path=None):
        if op=='video/cancel' and self.video and self.video.id==body.get('id'):
            self.video.cancelled.set()
        with self.lock:
            if op in ('document/open','document/save','preset-pack/open','preset-pack/save','video/begin') and not path:return {'cancelled':True}
            if op in ('document/open','preset-pack/open'):
                with Path(path).open('rb') as f:raw=f.read(8_000_001)
                if len(raw)>8_000_000:raise ValueError('声道文件超过 8 MB')
                document=client.invoke('preset-pack/validate' if op.startswith('preset-pack/') else 'document/validate',json.loads(raw.decode('utf-8-sig')))
                return {'document':document,'name':Path(path).name}
            if op in ('document/save','preset-pack/save'):
                document=client.invoke('preset-pack/create' if op.startswith('preset-pack/') else 'document/create',body)
                raw=json.dumps(document,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
                if len(raw)>8_000_000:raise ValueError('声道文件超过 8 MB')
                target=Path(path);temp=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.tmp')
                try:
                    with temp.open('xb') as f:f.write(raw)
                    temp.replace(target)
                finally:
                    if temp.exists():temp.unlink()
                return {'saved':True,'name':target.name}
            if op=='video/begin':
                if self.video and not self.video.closed:raise ValueError('已有视频正在导出')
                self.video=VideoSession(path,body);return {'id':self.video.id}
            if not self.video or self.video.id!=body.get('id'):raise ValueError('视频导出会话已失效')
            if op=='video/chunks':return self.video.append(body)
            if op=='video/finish':return self.video.finish()
            if op=='video/cancel':return self.video.cancel()
            raise ValueError('不支持的声道文件操作')

    def close(self):
        if self.video:self.video.cancelled.set()
        with self.lock:
            if self.video:self.video.cancel()
