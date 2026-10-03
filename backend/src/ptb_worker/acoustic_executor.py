"""One durable file task; heartbeat runs independently of scientific computation."""
import json
from ptb_worker.process_entry import command
import sys
import threading
from dataclasses import replace
from .managed_scratch import ManagedScratch
from .io.limits import Cancelled,LimitError,FormatError
from .acoustic_errors import public_error
from .segmentation import prepare_segments,unpack_bundle,SEGMENT_LIMITS,digest
from ptb_api.quota import StorageError,CHUNK_BYTES


def collect_scientific(entry, request, scratch, limits, stop, on_started=None, evidence=None, on_chunk=None):
    """Trusted fixed entry -> legacy bundle bytes, with platform-owned cleanup."""
    if sys.platform == 'linux':
        from .native.linux_runtime import run
        return run(entry, request, scratch.root, limits, stop=stop,
                   on_started=on_started, evidence=evidence, on_chunk=on_chunk)
    if sys.platform!='win32':raise FormatError('scientific_platform_unavailable')
    from .native.windows import InputPipe
    from .native.reaper import collect_pipe
    if entry in ('lpc', 'egg'):
        from .lpc_runtime import command as lpc_command
        from .egg_runtime import command as egg_command
        argv = (lpc_command if entry == 'lpc' else egg_command)(request, '')
    else:
        module = {'m07': 'ptb_worker.m07_child', 'm06': 'ptb_worker.m06_child', 'm08': 'ptb_worker.m08_child', 'acoustic': 'ptb_worker.science_child', 'segment': 'ptb_worker.segment_child',
                  'spec2wav': 'ptb_worker.spec2wav_child'}[entry]
        argv = command(module, str(request), '')
    pipe = InputPipe()
    argv[-1] = pipe.name
    raw, _ = collect_pipe(argv, pipe, scratch.root, limits, stop, on_started, on_chunk, evidence)
    return raw


def execute_acoustic_claim(store,claim,worker_id,stop,*,on_started=None,process_evidence=None):
    files=store.files;identity=(claim['id'],worker_id,claim['generation']);snapshot=json.loads(claim['snapshot'])
    # Offline EGG already permits 1 MiB reads/writes. Use that existing bound
    # instead of hundreds of tiny durable scratch transactions per interaction.
    # Hosted storage and other modules retain their established chunk sizes.
    local_large_chunks=snapshot['operation'] in ('egg_analysis','pitch_manipulation') and not store.postgres
    chunk_bytes=1_048_576 if local_large_chunks else CHUNK_BYTES
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
            for offset in range(0,item['size_bytes'],chunk_bytes):
                if abort.is_set() or stop.is_set():raise Cancelled('cancelled')
                raw.extend(files.read_input(identity,item['id'],offset,min(chunk_bytes,item['size_bytes']-offset)))
            if len(raw)!=item['size_bytes'] or digest(raw)!=item['sha256']:raise StorageError('input_unavailable',410)
            blobs[item['role']]=bytes(raw)
        progress[0]=.1
        with ManagedScratch(files,identity,write_chunk_bytes=chunk_bytes if local_large_chunks else 65536) as scratch:
            if snapshot['operation']=='pitch_manipulation' and snapshot.get('saved_copy'):
                result=dict(snapshot['copy_result'],name=snapshot['copy_name'])
                meta=dict(schema_version='m08/1',audio_sha256=snapshot['source_ref']['sha256'],results=[result])
                payloads=[(snapshot['copy_name'],blobs['audio']),('m08.ptb.json',json.dumps(meta,ensure_ascii=False,allow_nan=False).encode())]
            elif snapshot['operation']=='pitch_manipulation':
                from .m08_stream import Receiver
                audio_path=scratch.create(blobs['audio'],'.wav')
                native=files.output(identity,'m08-native.wav','temporary',64_000_000)
                try:
                    header=dict(config=snapshot['config']['analysis'],sha256=digest(blobs['audio']),
                        audio_path=str(audio_path),native_path=str(files.scratch_path(identity,native['id'])),
                        input_name=snapshot['input_assets'][0]['name'])
                    request=scratch.create(json.dumps(header).encode(),'.json')
                    receiver=Receiver(files,identity,header['sha256'],write_chunk_bytes=chunk_bytes if not store.postgres else 65536)
                    collect_scientific('m08',request,scratch,
                        replace(SEGMENT_LIMITS,timeout_seconds=120,process_bytes=1_000_000_000),
                        lambda:abort.is_set() or stop.is_set(),on_started,process_evidence,on_chunk=receiver.write)
                    receiver.finish();payloads=[]
                finally:files.release_scratch(identity,native['id'])
            elif snapshot['operation']=='lpc_analysis':
                from ptb_api.lpc_models import LPC_NAMES
                header=dict(config=snapshot['config']['analysis'],sha256=digest(blobs['audio']),
                    audio_size=len(blobs['audio']),textgrid_sha256=digest(blobs['textgrid']) if 'textgrid' in blobs else None,
                    input_name=next(i['name'] for i in snapshot['input_assets'] if i['role']=='audio'))
                request=scratch.create(json.dumps(header).encode()+b'\n'+blobs['audio']+blobs.get('textgrid',b''),'.json')
                raw=collect_scientific('lpc',request,scratch,
                    replace(SEGMENT_LIMITS,timeout_seconds=30,process_bytes=2_000_000_000,output_bytes=8_000_000),
                    lambda:abort.is_set() or stop.is_set(),on_started,process_evidence)
                bundle=unpack_bundle(raw,8_000_000)
                names=[f['name'] for f in bundle.manifest['files']]
                if (bundle.manifest.get('kind')!='prepared_lpc' or
                    bundle.manifest.get('audio_sha256')!=header['sha256'] or
                    bundle.manifest.get('textgrid_sha256')!=header['textgrid_sha256'] or sorted(names)!=sorted(LPC_NAMES)):
                    raise FormatError('source_mismatch')
                payloads=list(zip(names,bundle.payloads))
            elif snapshot['operation']=='egg_analysis':
                from ptb_api.egg_models import EggTaskConfig, expected_names
                header=dict(config=snapshot['config']['analysis'],sha256=digest(blobs['audio']),
                    input_name=next(i['name'] for i in snapshot['input_assets'] if i['role']=='audio'))
                native=None
                try:
                    if header['config'].get('keep_reaper_f0') and header['config']['mode']!='inverse':
                        from .egg_f0 import NATIVE_BYTES
                        from .acoustic_errors import AcousticFailure
                        if not getattr(files,'reaper_binary',None):raise AcousticFailure('egg_reaper_unavailable')
                        native=files.output(identity,'egg-native.wav','temporary',NATIVE_BYTES)
                        header.update(native_scratch=str(files.scratch_path(identity,native['id'])),reaper_binary=str(files.reaper_binary))
                    request=scratch.create(json.dumps(header).encode()+b'\n'+blobs['audio'],'.json')
                    raw=collect_scientific('egg',request,scratch,
                        replace(SEGMENT_LIMITS,timeout_seconds=240,process_bytes=3_000_000_000),
                        lambda:abort.is_set() or stop.is_set(),on_started,process_evidence)
                finally:
                    if native:files.release_scratch(identity,native['id'])
                bundle=unpack_bundle(raw,64_000_000)
                names=[f['name'] for f in bundle.manifest['files']]
                if (bundle.manifest.get('audio_sha256')!=header['sha256'] or
                    sorted(names)!=sorted(expected_names(EggTaskConfig.model_validate(header['config'])))):
                    raise FormatError('source_mismatch')
                payloads=list(zip(names,bundle.payloads))
            elif snapshot['operation']=='spectrogram_to_audio':
                header=dict(config=snapshot['config']['analysis'],sha256=digest(blobs['image']))
                request=scratch.create(json.dumps(header).encode()+b'\n'+blobs['image'],'.json')
                raw=collect_scientific('spec2wav',request,scratch,replace(SEGMENT_LIMITS,timeout_seconds=240),
                    lambda:abort.is_set() or stop.is_set(),on_started)
                bundle=unpack_bundle(raw,64_000_000)
                if bundle.manifest['image_sha256']!=header['sha256']:raise FormatError('source_mismatch')
                payloads=list(zip([f['name'] for f in bundle.manifest['files']],bundle.payloads))
            elif snapshot['operation']=='textgrid_segment':
                source=next(i for i in snapshot['input_assets'] if i['role']=='audio')
                bundle=prepare_segments(blobs['audio'],blobs['textgrid'],snapshot['layer'],scratch,audio_name=source['name'],
                    parent_result=blobs.get('parent_result'),legacy_result=blobs.get('legacy_result'),
                    legacy_name=next((i['name'] for i in snapshot['input_assets'] if i['role']=='legacy_result'),None),
                    stop=lambda:abort.is_set() or stop.is_set(),on_started=on_started)
                payloads=list(zip([f['name'] for f in bundle.manifest['files']],bundle.payloads))
                payloads.append(('segments.ptb.json',json.dumps(bundle.manifest,ensure_ascii=False,allow_nan=False).encode()))
            else:
                native=files.output(identity,'native-scratch.wav','temporary',16_000_000)
                try:
                    header=dict(request=dict(project_id=str(claim['project_id']),idempotency_key=claim['id'],
                        inputs={k:v for k,v in snapshot['input_refs'].items() if k not in ('parent_result','legacy_result')},config=snapshot['config']['analysis']),
                        inputs=snapshot['input_assets'],native_scratch=str(files.scratch_path(identity,native['id'])),
                        reaper_binary=str(files.reaper_binary))
                    request=scratch.create(json.dumps(header,ensure_ascii=False).encode()+b'\n'+b''.join(blobs[i['role']] for i in snapshot['input_assets']),'.json')
                    raw=collect_scientific('acoustic',request,scratch,
                        replace(SEGMENT_LIMITS,timeout_seconds=240),lambda:abort.is_set() or stop.is_set(),on_started,process_evidence)
                    bundle=unpack_bundle(raw,64_000_000)
                    if bundle.manifest['audio_sha256']!=digest(blobs['audio']):raise FormatError('source_mismatch')
                    payloads=list(zip([f['name'] for f in bundle.manifest['files']],bundle.payloads))
                finally:files.release_scratch(identity,native['id'])
        if sum(len(raw) for _,raw in payloads)>64_000_000:raise LimitError('output_budget_exceeded')
        progress[0]=.85
        for name,raw in payloads:
            if abort.is_set() or stop.is_set():raise Cancelled('cancelled')
            asset=files.output(identity,name,'result',len(raw))
            for offset in range(0,len(raw),chunk_bytes):files.write(identity,asset['id'],offset,raw[offset:offset+chunk_bytes])
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
