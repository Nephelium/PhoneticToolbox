"""One durable file task; heartbeat runs independently of scientific computation."""
import json
import sys
import threading
from dataclasses import replace
from .managed_scratch import ManagedScratch
from .io.limits import Cancelled,LimitError,FormatError
from .acoustic_errors import public_error
from .segmentation import prepare_segments,unpack_bundle,SEGMENT_LIMITS,digest
from ptb_api.quota import StorageError,CHUNK_BYTES


def execute_acoustic_claim(store,claim,worker_id,stop,*,on_started=None):
    files=store.files;identity=(claim['id'],worker_id,claim['generation']);snapshot=json.loads(claim['snapshot'])
    done=threading.Event();abort=threading.Event();errors=[];progress=[.01]
    def heartbeat():
        try:
            if stop.is_set():raise StorageError('cancelled',409)
            files.heartbeat(identity)
            if store.heartbeat(*identity,progress[0])!='running':raise StorageError('stale_worker',409)
        except Exception as exc:errors.append(getattr(exc,'code','heartbeat_failed'));abort.set()
    def pulse():
        while not done.is_set():
            heartbeat()
            if abort.is_set():return
            done.wait(min(.5,store.lease_seconds/3))
    timer=threading.Thread(target=pulse,daemon=True)
    try:
        heartbeat()
        if abort.is_set():raise Cancelled('cancelled')
        timer.start()
        blobs={}
        for item in snapshot['input_assets']:
            raw=bytearray()
            for offset in range(0,item['size_bytes'],CHUNK_BYTES):
                if abort.is_set() or stop.is_set():raise Cancelled('cancelled')
                raw.extend(files.read_input(identity,item['id'],offset,min(CHUNK_BYTES,item['size_bytes']-offset)))
            if len(raw)!=item['size_bytes'] or digest(raw)!=item['sha256']:raise StorageError('input_unavailable',410)
            blobs[item['role']]=bytes(raw)
        progress[0]=.1
        with ManagedScratch(files,identity) as scratch:
            if snapshot['operation']=='textgrid_segment':
                source=next(i for i in snapshot['input_assets'] if i['role']=='audio')
                bundle=prepare_segments(blobs['audio'],blobs['textgrid'],snapshot['layer'],scratch,audio_name=source['name'],
                    parent_result=blobs.get('parent_result'),legacy_result=blobs.get('legacy_result'),
                    legacy_name=next((i['name'] for i in snapshot['input_assets'] if i['role']=='legacy_result'),None),
                    stop=lambda:abort.is_set() or stop.is_set(),on_started=on_started)
                payloads=list(zip([f['name'] for f in bundle.manifest['files']],bundle.payloads))
                payloads.append(('segments.ptb.json',json.dumps(bundle.manifest,ensure_ascii=False,allow_nan=False).encode()))
            else:
                from .native.windows import InputPipe
                from .native.reaper import collect_pipe
                native=files.output(identity,'native-scratch.wav','temporary',16_000_000)
                try:
                    header=dict(request=dict(project_id=str(claim['project_id']),idempotency_key=claim['id'],
                        inputs={k:v for k,v in snapshot['input_refs'].items() if k not in ('parent_result','legacy_result')},config=snapshot['config']['analysis']),
                        inputs=snapshot['input_assets'],native_scratch=str(files.scratch_path(identity,native['id'])),
                        reaper_binary=str(files.reaper_binary))
                    request=scratch.create(json.dumps(header,ensure_ascii=False).encode()+b'\n'+b''.join(blobs[i['role']] for i in snapshot['input_assets']),'.json')
                    pipe=InputPipe()
                    raw,_=collect_pipe([sys.executable,'-B','-m','ptb_worker.science_child',str(request),pipe.name],pipe,
                        scratch.root,replace(SEGMENT_LIMITS,timeout_seconds=240),lambda:abort.is_set() or stop.is_set(),on_started)
                    bundle=unpack_bundle(raw,64_000_000)
                    if bundle.manifest['audio_sha256']!=digest(blobs['audio']):raise FormatError('source_mismatch')
                    payloads=list(zip([f['name'] for f in bundle.manifest['files']],bundle.payloads))
                finally:files.release_scratch(identity,native['id'])
        if sum(len(raw) for _,raw in payloads)>64_000_000:raise LimitError('output_budget_exceeded')
        progress[0]=.85
        for name,raw in payloads:
            if abort.is_set() or stop.is_set():raise Cancelled('cancelled')
            asset=files.output(identity,name,'result',len(raw))
            for offset in range(0,len(raw),CHUNK_BYTES):files.write(identity,asset['id'],offset,raw[offset:offset+CHUNK_BYTES])
            files.seal(identity,asset['id'])
        done.set();timer.join(timeout=6)
        if timer.is_alive() or abort.is_set() or stop.is_set():raise Cancelled('cancelled')
        files.complete(identity)
    except (Cancelled,LimitError,FormatError,StorageError,OSError,ValueError) as exc:
        code=errors[0] if errors else public_error(exc)
        files.fail(identity,code)
    finally:
        done.set()
        if timer.ident is not None:timer.join(timeout=6)
