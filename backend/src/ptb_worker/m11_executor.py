"""M11 formal worker: bounded external MFA, current fencing and atomic publication.

Local-only execution until P11/remote directory quota accounting is integrated.
Retained attempt directories are explicit local diagnostics, never server bypasses.
"""
import hashlib
import json
from pathlib import Path
import shutil
import threading
from uuid import uuid4
from .mfa.components import atomic_json,digest,no_links,safe_name
from .mfa.runtime import run,select,registry_root,TEMP_BYTES
from .store import canonical
from ptb_api.quota import CHUNK_BYTES
from phonetic_core.transcription.mfa_name_codec import encode_fs_name,decode_fs_name


def execute_claim(store,claim,worker_id,stop,*,evidence=None):
    files=store.files;identity=(claim['id'],worker_id,claim['generation'])
    snapshot=json.loads(claim['snapshot']);request=snapshot['request']
    done=threading.Event();abort=threading.Event();stage=['preparing'];resource=evidence if evidence is not None else {}
    root=registry_root()/'attempts'/f'{claim["id"]}-{claim["generation"]}-{uuid4().hex}'
    def pulse():
        while not done.is_set():
            try:
                files.heartbeat(identity)
                value={'preparing':.05,'runtime_checked':.1,'aligning':.25,'exporting':.8,'complete':.9}.get(stage[0],.1)
                if stop.is_set() or store.heartbeat(*identity,value)!='running':abort.set();return
            except Exception:abort.set();return
            done.wait(min(.5,store.lease_seconds/3))
    thread=threading.Thread(target=pulse,daemon=True)
    def cancelled():return stop.is_set() or abort.is_set()
    try:
        if store.postgres or snapshot.get('execution_route')!='desktop-local':raise ValueError('m11_waiting_verified_node')
        thread.start()
        runtime,model=select(request['runtime_id'],request['model_id'],stop=cancelled)
        if (runtime['fingerprint'],model['model_sha256'],model['dictionary_sha256'])!=(snapshot['runtime_fingerprint'],snapshot['model_sha256'],snapshot['dictionary_sha256']):raise ValueError('m11_component_changed')
        no_links(root);root.mkdir(parents=True)
        if shutil.disk_usage(root).free<TEMP_BYTES+64_000_000:raise ValueError('m11_temp_space')
        corpus=root/'corpus';corpus.mkdir()
        by_id={s['id']:s for s in snapshot['input_assets']}
        def copy_asset(asset_id,target):
            item=by_id[asset_id];h=hashlib.sha256();count=0
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as output:
                for offset in range(0,item['size_bytes'],CHUNK_BYTES):
                    if cancelled():raise ValueError('cancelled')
                    chunk=files.read_input(identity,asset_id,offset,min(CHUNK_BYTES,item['size_bytes']-offset))
                    output.write(chunk);h.update(chunk);count+=len(chunk)
            if h.hexdigest()!=item['sha256'] or count!=item['size_bytes']:raise ValueError('m11_input_changed')
        expected={}
        for index,item in enumerate(request['corpus']):
            name=str(safe_name(item['name']));relative=Path(name)
            encoded=Path(*(encode_fs_name(p) for p in relative.parts))
            copy_asset(item['audio']['asset_id'],corpus/encoded)
            copy_asset(item['transcript']['asset_id'],(corpus/encoded).with_suffix(item['transcript_format']))
            # Explicit index preserves duplicate basenames across speaker folders.
            expected[encoded.with_suffix('.TextGrid').as_posix()]=f'{index+1:03d}_{relative.stem}.TextGrid'
        model_path=root/'model.zip';dictionary=root/'dictionary.dict'
        shutil.copyfile(model['model'],model_path)
        if request.get('dictionary'):copy_asset(request['dictionary']['asset_id'],dictionary)
        else:shutil.copyfile(model['dictionary'],dictionary)
        if digest(model_path)!=snapshot['model_sha256']:raise ValueError('m11_model_changed')
        def progress(value):
            stage[0]=value
            with store.transaction() as tx:
                current=store._row(tx,claim['id'])
                if current and current['worker_id']==worker_id and current['generation']==claim['generation']:
                    store._event(tx,current,'m11_'+value,tx.now())
        result=run(runtime['path'],root,dict(action='align',model=str(model_path),dictionary=str(dictionary),
                   config=request['config'],expected_files=len(expected)),stop=cancelled,progress=progress,evidence=resource)
        outputs=[]
        for relative,name in expected.items():
            path=root/'output'/relative
            if not path.is_file():raise ValueError('m11_incomplete_output')
            if path.stat().st_size>8_000_000:raise ValueError('m11_output_budget')
            outputs.append((name,path.read_bytes()))
        provenance=dict(schema_version='m11/1',runtime=dict(id=runtime['id'],version=runtime['version'],fingerprint=runtime['fingerprint'],versions=result['versions']),
            model=dict(id=model['id'],sha256=digest(model_path)),dictionary_sha256=digest(dictionary),
            config=request['config'],inputs=request['corpus'],source_ids=['SRC-MFA','REF-MFA'],
            adaptation='ADR-M11-001',database='per-attempt-sqlite',resources=resource,
            textgrids=result['textgrids'],warnings=result.get('warnings',[]))
        outputs.append(('m11-provenance.json',json.dumps(provenance,ensure_ascii=False,allow_nan=False).encode()))
        if sum(len(raw) for _,raw in outputs)>64_000_000:raise ValueError('m11_output_budget')
        for name,raw in outputs:
            if cancelled():raise ValueError('cancelled')
            asset=files.output(identity,name,'result',len(raw))
            for offset in range(0,len(raw),CHUNK_BYTES):
                if cancelled():raise ValueError('cancelled')
                files.write(identity,asset['id'],offset,raw[offset:offset+CHUNK_BYTES])
            files.seal(identity,asset['id'])
        if cancelled():raise ValueError('cancelled')
        files.complete(identity)
    except Exception as exc:
        code=getattr(exc,'code',str(exc))
        if not (code.startswith('m11_') and len(code)<80) and code not in ('cancelled','quota_exceeded','output_budget_exceeded','input_unavailable'):code='m11_execution_failed'
        files.fail(identity,code)
        if root.exists():atomic_json(root/'failure.json',dict(error=code,resources=resource,partial_outputs_retained=True))
    finally:
        done.set()
        if thread.ident is not None:thread.join(timeout=6)
        if root.exists():atomic_json(root/'resources.json',resource)
