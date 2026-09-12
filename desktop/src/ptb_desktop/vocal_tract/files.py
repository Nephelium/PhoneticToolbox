"""Native selection and atomic files for M10, owned by one Qt window."""
import json
import threading
import uuid
from pathlib import Path
from .video import VideoSession


class VocalFiles:
    def __init__(self):self.video=None;self.lock=threading.RLock()

    def invoke(self,op,body,client,path=None):
        with self.lock:
            if op in ('document/open','document/save','video/begin') and not path:return {'cancelled':True}
            if op=='document/open':
                with Path(path).open('rb') as f:raw=f.read(8_000_001)
                if len(raw)>8_000_000:raise ValueError('关键帧文件超过 8 MB')
                document=client.invoke('document/validate',json.loads(raw.decode('utf-8-sig')))
                return {'document':document,'name':Path(path).name}
            if op=='document/save':
                document=client.invoke('document/create',body)
                raw=json.dumps(document,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
                if len(raw)>8_000_000:raise ValueError('关键帧文件超过 8 MB')
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
        with self.lock:
            if self.video:self.video.cancel()
