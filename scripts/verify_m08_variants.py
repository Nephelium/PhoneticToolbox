"""Owned real-Praat host for multi-variant Chrome and Qt acceptance."""
import base64
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from verification_artifacts import verification_run
from workbench_source import bind_sources

ROOT=Path(__file__).resolve().parents[1]
bind_sources(ROOT)


def fixtures(scratch):
    import numpy as np
    from scipy.io import wavfile
    from ptb_worker.local_acoustic_files import initialize_local_files
    inputs=scratch/'inputs';inputs.mkdir();saved=scratch/'saved';saved.mkdir()
    for name,hz in [('原音 ɑ̃˥',150),('原音 二',220)]:
        t=np.arange(16000)/16000
        wavfile.write(inputs/(name+'.wav'),16000,(.3*32767*np.sin(2*np.pi*hz*t)).astype('int16'))
    db=scratch/'jobs.sqlite3'
    # A new disposable database only; never migrate an existing workbench store.
    with sqlite3.connect(db) as dest:
        for migration in ('002_jobs_sqlite.sql','005_acoustic_batches_sqlite.sql'):
            dest.executescript((ROOT/'backend/migrations'/migration).read_text(encoding='utf8'))
    cache=scratch/'cache';cache.mkdir();initialize_local_files(cache)
    return inputs,saved,db,cache


def audit(scratch):
    import numpy as np
    import parselmouth
    results=[]
    for path in (scratch/'saved').glob('*.wav'):
        sound=parselmouth.Sound(str(path));f=sound.to_pitch().selected_array['frequency']
        results.append(dict(name=path.name,duration=sound.duration,f0=float(np.median(f[f>0])),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return results


def host():
    from ptb_desktop.file_provider import FileProvider
    from ptb_desktop.task_bridge import TaskBridge
    from ptb_desktop.local_service import LocalService
    with verification_run('m08-variants','chrome','Two 1s 16kHz PCM16 tones at 150/220 Hz; actual desktop M08Port/SQLite/Praat; isolated Chrome') as (out,scratch):
        inputs,saved,db,cache=fixtures(scratch)
        provider=FileProvider();directory=provider.choose('input',lambda:str(inputs));destination=provider.choose('output',lambda:str(saved))
        original={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        choices=0;output_mode='normal';fail_export=None
        with LocalService(db,local_files_root=cache) as service:
            bridge=TaskBridge(provider,service)
            print(json.dumps(dict(ready=True,out=str(out),scratch=str(scratch))),flush=True)
            for line in sys.stdin:
                request=json.loads(line);body=request['body']
                try:
                    if request['channel']=='test':
                        output_mode=body.get('output_mode','normal');fail_export=body.get('fail_export');value=dict(choices=choices,audit=audit(scratch))
                    elif request['channel']=='task':
                        if body.get('op')=='m08_export' and body.get('id')==fail_export:raise OSError('M08 test disk unavailable')
                        value=bridge.invoke(body)
                    else:
                        op=body['op']
                        if op=='hello':value=dict(kind='desktop',session=provider.session,api_version='1.1.0',tasks=True)
                        elif op=='fonts':value=['Arial','Microsoft YaHei','Doulos SIL']
                        elif op=='choose':
                            if body['purpose']=='output':choices+=1;value=None if output_mode=='cancel' else destination
                            else:value=directory
                        elif op=='list':value=provider.list(body.get('id') or directory['id'])
                        elif op=='read':
                            raw,sha=provider.read(body['id']);value=dict(base64=base64.b64encode(raw).decode(),sha256=sha)
                        else:raise ValueError('Unsupported test transport')
                    response=dict(id=request['id'],ok=True,value=value)
                except Exception as exc:response=dict(id=request['id'],ok=False,error=str(exc))
                print(json.dumps(response,ensure_ascii=False),flush=True)
        assert original=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        (out/'saved-audit.json').write_text(json.dumps(dict(choices=choices,files=audit(scratch),original_unchanged=True),ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':
    if '--host' in sys.argv:host()
    else:
        result=subprocess.run(['node',str(ROOT/'tests/e2e/m08-variants.cjs')],cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
        raise SystemExit(result.returncode)
