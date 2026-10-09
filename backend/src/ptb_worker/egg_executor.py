"""M03 stream input into owned scratch; publish complete bounded bundles."""
import hashlib
import json
import threading
from dataclasses import replace
from .managed_scratch import ManagedScratch
from .segmentation import SEGMENT_LIMITS, unpack_bundle
from .io.limits import Cancelled, FormatError
from .acoustic_errors import public_error


def execute_claim(store,claim,worker_id,stop,*,on_started=None,evidence=None):
    from .acoustic_executor import collect_scientific
    from ptb_api.egg_models import EggTaskConfig, expected_names
    files=store.files;identity=(claim['id'],worker_id,claim['generation']);snapshot=json.loads(claim['snapshot'])
    done=threading.Event();abort=threading.Event();progress=[.01];temporary=[]
    def cleanup():
        # A failed publication may already have removed every managed asset.
        for path in list(scratch.files):
            key=scratch.ids[path]
            with files.locked() as state:active=state['assets'].get(key,{}).get('state')!='deleted'
            if active:scratch.remove(path)
            else:scratch.used-=scratch.files.pop(path);scratch.ids.pop(path)
        while temporary:
            key=temporary.pop()
            with files.locked() as state:active=state['assets'].get(key,{}).get('state')!='deleted'
            if active:files.release_scratch(identity,key)
    def check():
        if stop.is_set() or abort.is_set():raise Cancelled('cancelled')
    def pulse():
        while not done.is_set():
            try:
                files.heartbeat(identity)
                if store.heartbeat(*identity,progress[0])!='running':abort.set();return
            except Exception:abort.set();return
            done.wait(min(.5,store.lease_seconds/3))
    timer=threading.Thread(target=pulse,daemon=True);scratch=ManagedScratch(files,identity,write_chunk_bytes=1048576)
    try:
        timer.start();source=next(i for i in snapshot['input_assets'] if i['role']=='audio')
        if source['size_bytes']>2_000_000_000:raise FormatError('egg_input_budget')
        slot=files.output(identity,'egg-source.wav','temporary',source['size_bytes']);temporary.append(slot['id'])
        digest=hashlib.sha256();offset=0
        while offset<source['size_bytes']:
            check();raw=files.read_input(identity,source['id'],offset,min(1048576,source['size_bytes']-offset))
            if not raw:raise FormatError('input_unavailable')
            digest.update(raw);files.write(identity,slot['id'],offset,raw);offset+=len(raw)
            progress[0]=.01+.09*offset/source['size_bytes']
        if digest.hexdigest()!=source['sha256']:raise FormatError('input_unavailable')
        header=dict(config=snapshot['config']['analysis'],sha256=source['sha256'],input_name=source['name'],
                    source_path=str(files.scratch_path(identity,slot['id'])))
        if header['config'].get('keep_reaper_f0') and header['config']['mode']!='inverse':
            from .egg_f0 import NATIVE_BYTES
            if not getattr(files,'reaper_binary',None):raise FormatError('egg_reaper_unavailable')
            native=files.output(identity,'egg-native.wav','temporary',NATIVE_BYTES);temporary.append(native['id'])
            header.update(native_scratch=str(files.scratch_path(identity,native['id'])),reaper_binary=str(files.reaper_binary))
        request=scratch.create(json.dumps(header).encode()+b'\n','.json')
        output_limit=256_000_000 if snapshot['config'].get('egg_bundle_revision')=='m03/2' else 64_000_000
        raw=collect_scientific('egg',request,scratch,
            replace(SEGMENT_LIMITS,timeout_seconds=1800,process_bytes=3_000_000_000,output_bytes=output_limit),
            lambda:abort.is_set() or stop.is_set(),on_started,evidence)
        check();bundle=unpack_bundle(raw,output_limit)
        names=[f['name'] for f in bundle.manifest['files']]
        if bundle.manifest.get('audio_sha256')!=source['sha256'] or sorted(names)!=sorted(expected_names(EggTaskConfig.model_validate(header['config']))):
            raise FormatError('source_mismatch')
        total=sum(len(p) for p in bundle.payloads);written=0
        for name,payload in zip(names,bundle.payloads):
            check();asset=files.output(identity,name,'result',len(payload))
            for offset in range(0,len(payload),1048576):
                check();block=payload[offset:offset+1048576];files.write(identity,asset['id'],offset,block)
                written+=len(block);progress[0]=.9+.099*written/max(total,1)
            files.seal(identity,asset['id'])
        check();cleanup()
        done.set();timer.join(timeout=6)
        if timer.is_alive():raise Cancelled('cancelled')
        check();files.complete(identity)
    except Exception as error:
        cleanup();files.fail(identity,public_error(error))
    finally:
        done.set()
        if timer.ident is not None:timer.join(timeout=6)
        cleanup()
