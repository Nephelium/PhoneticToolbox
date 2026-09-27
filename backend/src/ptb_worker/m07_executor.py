"""One complete M07 group per fenced shared task; no publication of partial WAVs."""
import hashlib
import json
import struct
import threading
from .managed_scratch import ManagedScratch
from .io.limits import Limits,Cancelled,FormatError,LimitError
from .m07_errors import public_error
from ptb_api.quota import StorageError,CHUNK_BYTES

LIMITS=Limits(input_bytes=49_000_000,output_bytes=64_000_000,process_bytes=1_000_000_000,timeout_seconds=120)

def unpack(raw,digest,request):
    if len(raw)<4 or len(raw)>LIMITS.output_bytes:raise FormatError('m07_invalid_bundle')
    size=struct.unpack('<I',raw[:4])[0]
    if not 0<size<=100000 or len(raw)<4+size:raise FormatError('m07_invalid_bundle')
    meta=json.loads(raw[4:4+size]);offset=4+size
    if meta.get('error'):raise FormatError(meta['error'])
    if meta.get('kind')!='prepared_m07' or meta.get('request_hash')!=digest:raise FormatError('m07_input_changed')
    expected=([f'step{i+1:02d}.wav' for i in range(request['generation']['step_count'])]+['combined_steps.wav'] if request['action']=='generate' else ['analysis.m07.json'])+['edited_f0.csv','m07.ptb.json']
    if [f['name'] for f in meta['files']]!=expected:raise FormatError('m07_incomplete_output')
    values=[]
    for file in meta['files']:
        if type(file['size'])!=int or not 0<file['size']<=LIMITS.output_bytes:raise FormatError('m07_invalid_bundle')
        value=raw[offset:offset+file['size']];offset+=file['size']
        if len(value)!=file['size'] or hashlib.sha256(value).hexdigest()!=file['sha256']:raise FormatError('m07_output_hash')
        values.append((file['name'],value))
    if offset!=len(raw):raise FormatError('m07_invalid_bundle')
    return values

def execute_claim(store,claim,worker_id,stop,*,on_started=None,evidence=None):
    files=store.files;identity=(claim['id'],worker_id,claim['generation']);snapshot=json.loads(claim['snapshot'])
    done=threading.Event();abort=threading.Event();errors=[]
    def heartbeat():
        files.heartbeat(identity)
        if stop.is_set() or store.heartbeat(*identity,.1)!='running':raise Cancelled('cancelled')
    def pulse():
        while not done.is_set():
            try:heartbeat()
            except Exception as exc:errors.append(public_error(exc));abort.set();return
            done.wait(min(.5,store.lease_seconds/3))
    thread=threading.Thread(target=pulse,daemon=True)
    try:
        heartbeat();thread.start();inputs={}
        for item in snapshot['input_assets']:
            raw=bytearray()
            for offset in range(0,item['size_bytes'],CHUNK_BYTES):
                if abort.is_set() or stop.is_set():raise Cancelled('cancelled')
                raw.extend(files.read_input(identity,item['id'],offset,min(CHUNK_BYTES,item['size_bytes']-offset)))
            if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('m07_input_changed')
            inputs[item['role']]=bytes(raw)
        with ManagedScratch(files,identity) as scratch:
            request={k:v for k,v in snapshot['request'].items() if k!='retry_of'}|{'idempotency_key':claim['id']}
            header=dict(request=request,request_hash=claim['request_hash'].strip(),inputs=snapshot['input_assets'])
            native=None
            try:
                if request['action']=='analyze' and request['analysis']['f0_backend']=='reaper':
                    native=files.output(identity,'m07-native.wav','temporary',400_000)
                    header.update(native_scratch=str(files.scratch_path(identity,native['id'])),reaper_binary=str(files.reaper_binary))
                packed=scratch.create(json.dumps(header,allow_nan=False).encode()+b'\n'+b''.join(inputs[item['role']] for item in snapshot['input_assets']),'.json')
                from .acoustic_executor import collect_scientific
                raw=collect_scientific('m07',packed,scratch,LIMITS,lambda:abort.is_set() or stop.is_set(),on_started,evidence)
                values=unpack(raw,header['request_hash'],request)
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
        if abort.is_set() or thread.is_alive():raise Cancelled('cancelled')
        files.complete(identity)
    except (ValueError,StorageError,OSError,Cancelled,LimitError) as exc:
        files.fail(identity,errors[0] if errors else public_error(exc))
    finally:
        done.set()
        if thread.ident is not None:thread.join(timeout=6)
