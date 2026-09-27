"""Chunked local input using the existing immutable asset state and disk budget."""
import base64
import hashlib
import os
import time
from uuid import uuid4
from .store import JobError

SUFFIXES=('.mp4','.mov','.avi','.mkv','.wmv','.m4v','.webm')

def begin(files,name,size):
    if not isinstance(name,str) or any(c in name for c in '/\\:\x00') or not name.lower().endswith(SUFFIXES) or not 0<size<=128_000_000:raise JobError('m05_invalid_video',422)
    with files.locked() as state:
        for item in state['assets'].values():
            if item.get('role')=='video' and item['state']=='uploading' and time.time()-item.get('touched_at',0)>3600:
                item.update(state='incomplete',reserved_bytes=0)
        files._save(state)
        # Abandoned partial uploads remain visible in storage accounting; bound
        # count so repeated renderer failures cannot grow metadata indefinitely.
        if sum(a['state']=='uploading' and a.get('role')=='video' for a in state['assets'].values())>=8:raise JobError('m05_pending_upload_limit',409)
        files._budget(size);key=str(uuid4())
        state['assets'][key]=dict(id=key,name=name,role='video',kind='input',state='uploading',sha256=None,size_bytes=0,expected_bytes=size,reserved_bytes=size,job_id=None,generation=0,touched_at=time.time())
        files._path(key).open('xb').close();files._save(state)
    return dict(id=key)

def block(files,key,offset,encoded):
    try:raw=base64.b64decode(encoded,validate=True)
    except Exception:raise JobError('m05_invalid_block',422) from None
    if not 0<len(raw)<=262144:raise JobError('m05_invalid_block',422)
    with files.locked() as state:
        item=files._asset(key)
        if item['state']!='uploading' or item.get('role')!='video' or item['size_bytes']!=offset or offset+len(raw)>item['expected_bytes']:raise JobError('m05_upload_state',409)
        with files._path(key).open('ab') as stream:
            if stream.tell()!=offset:raise JobError('m05_upload_state',409)
            if stream.write(raw)!=len(raw):raise OSError('short_write')
            stream.flush();os.fsync(stream.fileno())
        item['size_bytes']+=len(raw);item['touched_at']=time.time();files._save(state)
    return dict(size=item['size_bytes'])

def finish(files,key):
    with files.locked() as state:
        item=files._asset(key)
        if item.get('role')!='video' or item['state']!='uploading' or item['size_bytes']!=item['expected_bytes']:raise JobError('m05_upload_incomplete',409)
        h=hashlib.sha256()
        with files._path(key).open('rb') as stream:
            for raw in iter(lambda:stream.read(262144),b''):h.update(raw)
        item.update(state='ready',reserved_bytes=0,sha256=h.hexdigest());files._save(state)
    return dict(asset_id=key,sha256=h.hexdigest())

def abort(files,key):
    with files.locked() as state:
        item=files._asset(key)
        if item.get('role')!='video' or item['state'] not in ('uploading','incomplete'):raise JobError('m05_upload_state',409)
        item.update(state='incomplete',reserved_bytes=0);files._save(state)
    return dict(aborted=True)
