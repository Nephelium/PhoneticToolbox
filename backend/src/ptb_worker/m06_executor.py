"""M06 uses the public bounded process adapter and fenced quota writer."""
import hashlib
import json
import struct
import threading
from .managed_scratch import ManagedScratch
from .io.limits import Limits,Cancelled,FormatError,LimitError
from ptb_api.quota import StorageError,CHUNK_BYTES

LIMITS=Limits(input_bytes=16_000_000,output_bytes=24_000_000,process_bytes=1_000_000_000,timeout_seconds=120)


def unpack(raw,sha,action):
    if len(raw)<4 or len(raw)>LIMITS.output_bytes:raise FormatError('m06_invalid_bundle')
    size=struct.unpack('<I',raw[:4])[0]
    if not 0<size<=100000 or len(raw)<4+size:raise FormatError('m06_invalid_bundle')
    meta=json.loads(raw[4:4+size]);offset=4+size
    if meta.get('error'):raise FormatError(meta['error'])
    if meta.get('kind')!='prepared_m06' or meta.get('input_sha256')!=sha:raise FormatError('m06_input_changed')
    expected=(['synthesis.wav'] if action in ('synthesize','resynthesize') else [])+['m06.ptb.json','parameters.csv']+(['analysis.npz'] if action=='resynthesize' else [])
    if [f['name'] for f in meta['files']]!=expected:raise FormatError('m06_incomplete_output')
    values=[]
    for f in meta['files']:
        if type(f['size'])!=int or not 0<f['size']<=LIMITS.output_bytes:raise FormatError('m06_invalid_bundle')
        value=raw[offset:offset+f['size']];offset+=f['size']
        if len(value)!=f['size'] or hashlib.sha256(value).hexdigest()!=f['sha256']:raise FormatError('m06_output_hash')
        values.append((f['name'],value))
    if offset!=len(raw):raise FormatError('m06_invalid_bundle')
    return values


def execute_claim(store,claim,worker_id,stop,*,on_started=None,evidence=None):
    files=store.files;identity=(claim['id'],worker_id,claim['generation']);snapshot=json.loads(claim['snapshot'])
    done=threading.Event();abort=threading.Event()
    def heartbeat():
        files.heartbeat(identity)
        if stop.is_set() or store.heartbeat(*identity,.1)!='running':raise Cancelled('cancelled')
    def pulse():
        while not done.is_set():
            try:heartbeat()
            except Exception:abort.set();return
            done.wait(min(.5,store.lease_seconds/3))
    thread=threading.Thread(target=pulse,daemon=True)
    try:
        heartbeat();thread.start();inputs={}
        for item in snapshot['input_assets']:
            data=bytearray()
            for offset in range(0,item['size_bytes'],CHUNK_BYTES):
                if abort.is_set() or stop.is_set():raise Cancelled('cancelled')
                data.extend(files.read_input(identity,item['id'],offset,min(CHUNK_BYTES,item['size_bytes']-offset)))
            if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('m06_input_changed')
            inputs[item['role']]=bytes(data)
        raw=inputs.get('audio',b'');sha=hashlib.sha256(raw).hexdigest();config=snapshot['config']
        with ManagedScratch(files,identity) as scratch:
            header=dict(parameters=inputs['table'].decode('utf8'),action=config['action'],seed=config['seed'],input_sha256=sha)
            from phonetic_core.synthesis.klatt.api import import_parameters
            parameters=import_parameters(header['parameters']);native=None
            try:
                if header['action'] in ('extract','resynthesize') and parameters['f0_method']=='reaper':
                    from pathlib import Path
                    from .native.reaper import REAPER_SHA256
                    binary=Path(getattr(files,'reaper_binary',None) or '')
                    if not binary.is_file() or binary.stat().st_size>16_000_000 or hashlib.sha256(binary.read_bytes()).hexdigest()!=REAPER_SHA256:
                        raise ValueError('m06_reaper_unavailable')
                    native=files.output(identity,'m06-native.wav','temporary',400_000)
                    header.update(native_scratch=str(files.scratch_path(identity,native['id'])),reaper_binary=str(binary))
                request=scratch.create(json.dumps(header,allow_nan=False).encode()+b'\n'+raw,'.json')
                from .acoustic_executor import collect_scientific
                bundle=collect_scientific('m06',request,scratch,LIMITS,lambda:abort.is_set() or stop.is_set(),on_started,evidence)
                values=unpack(bundle,sha,header['action'])
            finally:
                if native:files.release_scratch(identity,native['id'])
        for name,value in values:
            if abort.is_set() or stop.is_set():raise Cancelled('cancelled')
            asset=files.output(identity,name,'result',len(value))
            for offset in range(0,len(value),CHUNK_BYTES):
                if abort.is_set() or stop.is_set():raise Cancelled('cancelled')
                files.write(identity,asset['id'],offset,value[offset:offset+CHUNK_BYTES])
            files.seal(identity,asset['id'])
        done.set();thread.join(timeout=6);heartbeat()
        if abort.is_set():raise Cancelled('cancelled')
        files.complete(identity)
    except (ValueError,StorageError,OSError,Cancelled,LimitError) as exc:
        code=getattr(exc,'code',str(exc))
        if code=='native_timeout':code='m06_timeout'
        files.fail(identity,code if code.startswith('m06_') or code in ('cancelled','quota_exceeded','output_budget_exceeded') else 'execution_failed')
    finally:
        done.set()
        if thread.ident is not None:thread.join(timeout=6)
