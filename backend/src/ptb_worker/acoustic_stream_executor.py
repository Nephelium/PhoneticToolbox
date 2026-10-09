"""M01 bounded local task, native Job ownership and durable progress."""
import json
import os
import time
import threading
import hashlib
import shutil
from pathlib import Path
from uuid import uuid4
from .acoustic_stream_child import atomic
from .io.scratch import no_links

def execute_claim(store,claim,worker_id,stop,*,evidence=None,on_started=None):
    from .native.windows import OwnedProcess
    from .process_entry import command
    files=store.files;identity=(claim['id'],worker_id,claim['generation']);snapshot=json.loads(claim['snapshot'])
    # Keep native REAPER's input paths under its legacy Windows path bound.
    # The request/resources retain the job identity; directory identity is random.
    root=Path(files.scratch_root)/'.m1'/uuid4().hex[:16]
    no_links(root);root.mkdir(parents=True)
    done=threading.Event();abort=threading.Event();state=[.005];resource=evidence if evidence is not None else {}
    process=None;started=time.monotonic()
    def pulse():
        while not done.is_set():
            try:
                files.heartbeat(identity)
                if stop.is_set() or store.heartbeat(*identity,state[0])!='running':abort.set();return
            except Exception:abort.set();return
            done.wait(min(.5,store.lease_seconds/3))
    heartbeat=threading.Thread(target=pulse,daemon=True)
    def check():
        if stop.is_set() or abort.is_set():
            from .io.limits import Cancelled
            raise Cancelled('cancelled')
    try:
        if store.postgres or os.name!='nt':raise ValueError('m01_runtime_not_admitted')
        if shutil.disk_usage(root).free<sum(a['size_bytes'] for a in snapshot['input_assets'])+1_000_000_000:
            from ptb_api.quota import StorageError
            raise StorageError('disk_space_low',507)
        heartbeat.start()
        for item in snapshot['input_assets']:
            check();name='input.wav' if item['role']=='audio' else item['role']+'.bin';h=hashlib.sha256()
            with (root/name).open('xb') as output:
                for offset in range(0,item['size_bytes'],1048576):
                    check();raw=files.read_input(identity,item['id'],offset,min(1048576,item['size_bytes']-offset))
                    if not raw:raise ValueError('input_unavailable')
                    output.write(raw);h.update(raw)
            if h.hexdigest()!=item['sha256']:raise ValueError('input_unavailable')
        source=next(i for i in snapshot['input_assets'] if i['role']=='audio')
        atomic(root/'request.json',dict(config=snapshot['config']['analysis'],index=snapshot['batch_index'],
            operation=snapshot['operation'],layer=snapshot.get('layer'),audio_name=source['name'],
            audio_sha256=source['sha256'],reaper_binary=str(files.reaper_binary)))
        process=OwnedProcess(command('ptb_worker.acoustic_stream_child',str(root)),root,2_147_483_648)
        if on_started:on_started(process.info.pid)
        last_check=0
        while process.poll() is None:
            check()
            if time.monotonic()-started>21600:raise ValueError('deadline_exceeded')
            if (root/'status.json').exists():
                value=json.loads((root/'status.json').read_text('utf-8'));state[0]=max(state[0],min(.94,float(value['progress'])))
            if time.monotonic()-last_check>3:
                last_check=time.monotonic()
                if sum(p.stat().st_size for p in root.iterdir() if p.is_file())>6_000_000_000:raise ValueError('m01_output_budget')
                if shutil.disk_usage(root).free<512_000_000:
                    from ptb_api.quota import StorageError
                    raise StorageError('disk_space_low',507)
            time.sleep(.2)
        response=root/'response.json'
        if not response.exists():raise ValueError('m01_process_failed')
        value=json.loads(response.read_text('utf-8'))
        if not value.get('success'):raise ValueError(value.get('error','m01_process_failed'))
        names=['result.xlsx','result.ptb.sqlite','result.ptb.json']
        if snapshot['operation']=='textgrid_segment':
            names=value['files']
            if not 1<=len(names)<=3001 or len(set(names))!=len(names) or 'segments.ptb.json' not in names or any(Path(n).name!=n or '/' in n or '\\' in n or not n.endswith(('.wav','.xlsx','.ptb.sqlite','.ptb.json')) for n in names):raise ValueError('m01_invalid_output')
        elif value['files']!=names:raise ValueError('m01_invalid_output')
        total=sum((root/n).stat().st_size for n in names);written=0
        if total>5_000_000_000:raise ValueError('m01_output_budget')
        for name in names:
            path=root/name;no_links(path);asset=files.output(identity,name,'result',path.stat().st_size)
            with path.open('rb') as stream:
                offset=0
                for raw in iter(lambda:stream.read(1048576),b''):
                    check();files.write(identity,asset['id'],offset,raw);offset+=len(raw);written+=len(raw)
                    state[0]=.94+.059*written/total
            files.seal(identity,asset['id'])
        check();files.complete(identity)
    except Exception as error:
        from .acoustic_errors import public_error
        files.fail(identity,public_error(error))
        atomic(root/'failure.json',dict(error=public_error(error),type=type(error).__name__))
    finally:
        if process:
            resource.update(memory_peak_bytes=process.memory_peak(),wall_seconds=time.monotonic()-started)
            process.close();resource['group_cleaned']=process.group_cleaned
        done.set()
        if heartbeat.ident is not None:heartbeat.join(timeout=6)
        # These files were created solely by this attempt. Managed result assets
        # and original recordings are outside this UUID directory and untouched.
        if root.resolve().parent==(Path(files.scratch_root)/'.m1').resolve():
            for child in root.iterdir():
                no_links(child)
                if child.is_file() and child.suffix not in ('.json','.next'):child.unlink()
                elif child.is_dir() and child.name.startswith('m01-'):
                    for leaf in child.rglob('*'):no_links(leaf)
                    shutil.rmtree(child)
            resource['temporary_payloads_cleaned']=True
        atomic(root/'resources.json',resource)
