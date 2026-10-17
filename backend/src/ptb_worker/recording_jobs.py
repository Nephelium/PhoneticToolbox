"""M16 uses existing durable queue, quota, fencing, cancellation and cleanup."""
import json
import hashlib
import struct
import threading
from pathlib import Path
import sys
from ptb_api.recording_models import RecordingRequest
from ptb_api.quota import StorageError,CHUNK_BYTES
from .store import JobError
from .managed_scratch import ManagedScratch
from .io.limits import Limits, Cancelled


def submit(store,owner,body,*,retry_of=None):
    body=RecordingRequest.model_validate(body)
    if not store.postgres or not store.files or store.files.storage.workspaces is None or sys.platform!='linux':raise JobError('recording_server_unavailable',503)
    from .native.linux_runtime import load_profile
    load_profile()
    from .native.capabilities import linux_capabilities
    if 'recording_audio' not in linux_capabilities(store)[0]:raise JobError('recording_server_unavailable',503)
    return store.files.submit(owner,body,retry_of=retry_of)


def execute(store,claim,worker,stop):
    files=store.files;identity=(claim['id'],worker,claim['generation']);done=threading.Event();abort=threading.Event();progress=[.01]
    def pulse():
        while not done.is_set():
            try:
                files.heartbeat(identity)
                if store.heartbeat(*identity,progress[0])!='running':abort.set();return
            except Exception:abort.set();return
            done.wait(.5)
    timer=threading.Thread(target=pulse,daemon=True)
    try:
        timer.start();snapshot=json.loads(claim['snapshot']);config=snapshot['config'];blobs=[]
        with files.storage._locked() as conn:_,inputs=files._fence(conn,identity)
        for item in inputs:
            if item['size_bytes']>100_000_000:raise StorageError('output_budget_exceeded',413)
            raw=bytearray()
            for offset in range(0,item['size_bytes'],CHUNK_BYTES):
                if abort.is_set() or stop.is_set():raise Cancelled('cancelled')
                raw.extend(files.read_input(identity,str(item['id']),offset,min(CHUNK_BYTES,item['size_bytes']-offset)))
            if hashlib.sha256(raw).hexdigest()!=item['input_sha256']:raise StorageError('input_unavailable',410)
            blobs.append(bytes(raw))
        progress[0]=.15
        # Reserve quota before native execution, including managed scratch.
        import io,soundfile as sf
        info=sf.info(io.BytesIO(blobs[0]));estimated=info.frames*info.channels*4+1024
        if config['action']=='paste':
            reference=sf.info(io.BytesIO(blobs[-1]));estimated+=reference.frames*reference.channels*4
        output=None if config['action']=='noise' else files.output(identity,'录音-'+config['action']+'.wav','result',min(100_000_000,estimated))
        metadata=files.output(identity,'录音处理.json','result',4096)
        with ManagedScratch(files,identity,budget=201_000_000) as scratch:
            paths=[str(scratch.create(b,'.wav')) for b in blobs]
            request=scratch.create(json.dumps({'paths':paths,'config':config}).encode(),'.json')
            from .native.linux_runtime import run
            raw=run('recording',request,scratch.root,Limits(input_bytes=1,output_bytes=100_010_000,process_bytes=800_000_000,timeout_seconds=120),stop=lambda:stop.is_set() or abort.is_set())
        if stop.is_set() or abort.is_set():raise Cancelled('cancelled')
        if len(raw)<4:raise ValueError('invalid output')
        n=struct.unpack_from('<I',raw)[0]
        if not 0<n<=10000 or 4+n>len(raw):raise ValueError('invalid output')
        result=json.loads(raw[4:4+n])
        if not result['ok']:
            # The message is bounded scientific validation text, not a traceback/path.
            safe={'error':result['error']};result['metadata']=safe
            raise StorageError('recording_invalid_input',422)
        progress[0]=.9
        if output:
            wav=raw[4+n:]
            if not wav.startswith(b'RIFF'):raise ValueError('invalid WAV output')
            resize(files,identity,output['id'],len(wav))
            for offset in range(0,len(wav),CHUNK_BYTES):files.write(identity,output['id'],offset,wav[offset:offset+CHUNK_BYTES])
            files.seal(identity,output['id'])
        text=json.dumps(result['metadata'],ensure_ascii=False,allow_nan=False).encode()
        resize(files,identity,metadata['id'],len(text))
        files.write(identity,metadata['id'],0,text);files.seal(identity,metadata['id']);files.complete(identity)
    except Cancelled:files.fail(identity,'cancelled')
    except StorageError as exc:files.fail(identity,exc.code)
    except Exception:files.fail(identity,'execution_failed')
    finally:done.set();timer.join(timeout=3)


def resize(files,identity,asset_id,size):
    """Set exact output length inside its already reserved upper bound."""
    with files.storage._locked() as conn:
        files._fence(conn,identity);row=files._output(conn,identity,asset_id)
        if not 0<size<=row['expected_bytes'] or row['size_bytes']:raise StorageError('output_budget_exceeded',413)
        conn.execute('UPDATE ptb_storage.assets SET expected_bytes=%s WHERE id=%s',(size,asset_id))
