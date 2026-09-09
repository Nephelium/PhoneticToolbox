"""M01-E HTTP/UI test double. No PG/persistence/quota or algorithm evidence."""
from pathlib import Path
import hashlib
import time
from uuid import uuid4
from ptb_api.quota import StorageError

GRID=b'File type = "ooTextFile short"\n"TextGrid"\n0\n.8\n<exists>\n1\n"IntervalTier"\n"IPA"\n0\n.8\n2\n0\n.4\n"a"\n.4\n.8\n"i"\n'


class PreviewStorage:
    ready=True
    def __init__(self):self.assets={};self.payloads={};self.owners={}
    def add(self,owner,project,name,payload):
        key=str(uuid4());self.payloads[key]=payload;self.owners[key]=owner
        self.assets[key]=dict(id=key,project_id=project,name=name,kind='input',state='ready',size_bytes=len(payload),reserved_bytes=0,
            expected_bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),created_at=time.time(),expires_at=time.time()+3600,error_code=None)
        return key
    def metadata(self,owner,key):
        key=str(key)
        if self.owners.get(key)!=owner:raise StorageError('asset_not_found',404)
        if self.assets[key]['expires_at']<=time.time():raise StorageError('asset_expired',410)
        return self.assets[key].copy()
    def list(self,owner,project,order):
        return [self.metadata(owner,k) for k,a in self.assets.items() if self.owners[k]==owner and a['project_id']==str(project)]
    def read_block(self,owner,key,offset,size):self.metadata(owner,key);return self.payloads[str(key)][offset:offset+size]
    def usage(self,owner):
        used=sum(a['size_bytes'] for k,a in self.assets.items() if self.owners[k]==owner)
        return dict(quota_bytes=5_000_000_000,used_bytes=used,reserved_bytes=0,available_bytes=5_000_000_000-used,frozen=False,ready=True)
