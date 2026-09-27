"""Local M05 streaming publication under the shared lease and Windows Job Object."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from uuid import uuid4
from .io.scratch import no_links

def run_child(root,request,stop,progress,evidence):
    from .native.windows import OwnedProcess
    from .m05_child import atomic
    runtime=Path(os.environ['PTB_M05_PYTHON']).absolute();no_links(runtime)
    atomic(root/'request.json',request)
    process=None;started=time.monotonic();peak_disk=0
    try:
        if stop():raise ValueError('cancelled')
        process=OwnedProcess([str(runtime),'-I','-B',str(Path(__file__).with_name('m05_child.py')),str(root/'request.json')],root,2_147_483_648)
        while True:
            if stop():raise ValueError('cancelled')
            if time.monotonic()-started>1800:raise ValueError('m05_timeout')
            size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());peak_disk=max(size,peak_disk)
            if size>700_000_000:raise ValueError('m05_disk_budget')
            status=root/'status.json'
            if status.exists():progress(json.loads(status.read_text('utf-8')))
            code=process.poll()
            if code is not None:
                if code:raise ValueError('m05_process_crashed')
                path=root/'response.json'
                if not path.is_file() or path.stat().st_size>65536:raise ValueError('m05_response_missing')
                response=json.loads(path.read_text('utf-8'))
                if not response.get('success'):
                    if response.get('type')=='MemoryError':raise ValueError('m05_memory_budget')
                    code='m05_'+response.get('error','execution_failed')
                    from .m05_errors import M05_ERRORS
                    raise ValueError(code if code in M05_ERRORS else 'm05_execution_failed')
                return response
            time.sleep(.1)
    finally:
        if process:
            evidence.update(memory_peak_bytes=process.memory_peak(),temp_peak_sampled_bytes=peak_disk,wall_seconds=time.monotonic()-started,process_budget_bytes=2_147_483_648)
            process.close();evidence['group_cleaned']=process.group_cleaned

def execute_claim(store,claim,worker_id,stop,*,evidence=None):
    from .m05_child import atomic
    files=store.files;identity=(claim['id'],worker_id,claim['generation'])
    snapshot=json.loads(claim['snapshot']);request=snapshot['request'];resource=evidence if evidence is not None else {}
    root=Path(files.scratch_root)/'m05-attempts'/f'{claim["id"]}-{claim["generation"]}-{uuid4().hex}'
    done=threading.Event();abort=threading.Event()
    def cancelled():return stop.is_set() or abort.is_set()
    def pulse():
        while not done.is_set():
            try:
                files.heartbeat(identity)
                if stop.is_set() or store.heartbeat(*identity,.1)!='running':abort.set();return
            except Exception:abort.set();return
            done.wait(min(.5,store.lease_seconds/3))
    heartbeat=threading.Thread(target=pulse,daemon=True)
    try:
        if store.postgres or os.name!='nt':raise ValueError('m05_runtime_not_admitted')
        no_links(root);root.mkdir(parents=True)
        if shutil.disk_usage(root).free<1_500_000_000:raise ValueError('m05_disk_space')
        heartbeat.start();source=snapshot['input_assets'][0];input_name='input'+Path(source['name']).suffix
        h=hashlib.sha256()
        with (root/input_name).open('xb') as output:
            for offset in range(0,source['size_bytes'],262144):
                if cancelled():raise ValueError('cancelled')
                raw=files.read_input(identity,source['id'],offset,min(262144,source['size_bytes']-offset))
                if not raw:raise ValueError('m05_incomplete_input')
                output.write(raw);h.update(raw)
        if h.hexdigest()!=source['sha256']:raise ValueError('m05_input_changed')
        last=[-1]
        def progress(value):
            count=value.get('frames',0)
            if count!=last[0]:
                last[0]=count
                with store.transaction() as tx:
                    current=store._row(tx,claim['id'])
                    if current and current['worker_id']==worker_id and current['generation']==claim['generation']:store._event(tx,current,'m05_frames_'+str(count),tx.now())
        result=run_child(root,dict(input=input_name,original_name=source['name'],config=request['config']),cancelled,progress,resource)
        atomic(root/'results'/'resources.json',resource);result['files'].append('resources.json')
        if sum((root/'results'/n).stat().st_size for n in result['files'])>512_000_000:raise ValueError('m05_output_budget')
        for name in result['files']:
            if cancelled():raise ValueError('cancelled')
            if Path(name).name!=name:raise ValueError('m05_unsafe_output')
            path=root/'results'/name;no_links(path)
            asset=files.output(identity,name,'result',path.stat().st_size)
            with path.open('rb') as stream:
                offset=0
                for raw in iter(lambda:stream.read(262144),b''):
                    if cancelled():raise ValueError('cancelled')
                    files.write(identity,asset['id'],offset,raw);offset+=len(raw)
            files.seal(identity,asset['id'])
        if cancelled():raise ValueError('cancelled')
        files.complete(identity)
    except Exception as error:
        code=getattr(error,'code',str(error))
        if not (code.startswith('m05_') and len(code)<100) and code not in ('cancelled','quota_exceeded','output_budget_exceeded','input_unavailable'):code='m05_execution_failed'
        files.fail(identity,code)
        if root.exists():
            import traceback
            atomic(root/'failure.json',dict(error=code,exception_type=type(error).__name__,diagnostic=str(error)[:1000],traceback=traceback.format_exc(limit=5),resources=resource,partial_outputs_retained=True))
    finally:
        done.set()
        if heartbeat.ident is not None:heartbeat.join(timeout=6)
