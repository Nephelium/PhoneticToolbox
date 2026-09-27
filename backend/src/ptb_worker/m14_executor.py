"""Module handler using the existing P11 Windows pipe/Job Object implementation.

No resource/process implementation is duplicated or changed. Linux stays closed
until P11 registers the fixed child and proves its resource gate.
"""
import hashlib
import json
import struct
import sys
import threading
from .managed_scratch import ManagedScratch
from .io.limits import Limits,Cancelled,FormatError,LimitError
from ptb_api.quota import StorageError,CHUNK_BYTES
from phonetic_core.transcription.phonology.models import NAMES

LIMITS=Limits(input_bytes=2_000_000,output_bytes=16_000_000,process_bytes=512_000_000,timeout_seconds=60)


def unpack(raw,sha,action):
    if len(raw)<4 or len(raw)>16_000_000:raise FormatError('m14_invalid_bundle')
    size=struct.unpack('<I',raw[:4])[0]
    if not 0<size<=100_000 or len(raw)<4+size:raise FormatError('m14_invalid_bundle')
    meta=json.loads(raw[4:4+size]);offset=4+size
    if meta.get('error'):raise FormatError(meta['error'])
    if meta.get('kind')!='prepared_m14' or meta.get('input_sha256')!=sha:raise FormatError('m14_input_changed')
    expected=list(NAMES) if action=='export' else ['m14-preview.json']
    if [f['name'] for f in meta['files']]!=expected:raise FormatError('m14_incomplete_output')
    values=[]
    for f in meta['files']:
        if type(f['size'])!=int or not 0<f['size']<=16_000_000:raise FormatError('m14_invalid_bundle')
        value=raw[offset:offset+f['size']];offset+=f['size']
        if len(value)!=f['size'] or hashlib.sha256(value).hexdigest()!=f['sha256']:raise FormatError('m14_output_hash')
        values.append((f['name'],value))
    if offset!=len(raw):raise FormatError('m14_invalid_bundle')
    return values


def execute_claim(store,claim,worker_id,stop,*,on_started=None,evidence=None):
    files=store.files;identity=(claim['id'],worker_id,claim['generation']);snapshot=json.loads(claim['snapshot'])
    done=threading.Event();abort=threading.Event();errors=[]
    def pulse():
        while not done.is_set():
            try:
                files.heartbeat(identity)
                if stop.is_set() or store.heartbeat(*identity,.1)!='running':abort.set();return
            except Exception:errors.append('stale_worker');abort.set();return
            done.wait(min(.5,store.lease_seconds/3))
    thread=threading.Thread(target=pulse,daemon=True)
    try:
        if sys.platform!='win32':raise ValueError('m14_runtime_unavailable')
        thread.start();item=snapshot['input_assets'][0]
        raw=b''.join(files.read_input(identity,item['id'],offset,min(CHUNK_BYTES,item['size_bytes']-offset)) for offset in range(0,item['size_bytes'],CHUNK_BYTES))
        sha=hashlib.sha256(raw).hexdigest()
        if sha!=item['sha256'] or len(raw)!=item['size_bytes']:raise ValueError('m14_input_changed')
        with ManagedScratch(files,identity) as scratch:
            header=dict(config=snapshot['config']['analysis'],name=item['name'],sha256=sha)
            request=scratch.create(json.dumps(header).encode()+b'\n'+raw,'.json')
            from .native.windows import InputPipe
            from .native.reaper import collect_pipe
            from .process_entry import command
            pipe=InputPipe()
            bundle,_=collect_pipe(command('ptb_worker.m14_child',str(request),pipe.name),pipe,scratch.root,LIMITS,lambda:abort.is_set() or stop.is_set(),on_started,None,evidence)
            values=unpack(bundle,sha,header['config']['action'])
        for name,value in values:
            if abort.is_set() or stop.is_set():raise Cancelled('cancelled')
            asset=files.output(identity,name,'result',len(value))
            for offset in range(0,len(value),CHUNK_BYTES):files.write(identity,asset['id'],offset,value[offset:offset+CHUNK_BYTES])
            files.seal(identity,asset['id'])
        done.set();thread.join(timeout=6)
        if abort.is_set() or stop.is_set():raise Cancelled('cancelled')
        files.complete(identity)
    except (ValueError,StorageError,OSError,Cancelled,LimitError) as e:
        code=getattr(e,'code',str(e))
        if code=='native_timeout':code='m14_timeout'
        files.fail(identity,code if code.startswith('m14_') or code=='cancelled' else 'execution_failed')
    finally:
        done.set()
        if thread.ident is not None:thread.join(timeout=6)
