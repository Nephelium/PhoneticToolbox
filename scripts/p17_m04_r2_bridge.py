"""Owned P17 real-recording bridge. Creates only a new test workspace.

Audio is copied byte-for-byte from the authorized corpus. No audio generation.
"""
import base64
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/p) for p in ('desktop/src','backend/src','packages/phonetic_core/src')]
os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/p) for p in ('desktop/src','backend/src','packages/phonetic_core/src'))
from phonetic_core.textgrid import parse_textgrid
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.local_service import LocalService
from ptb_desktop.task_bridge import TaskBridge
from ptb_worker.local_workspace import prepare_workspace

CORPUS=Path(r'C:\Users\13680\Desktop\project\音频数据')

def prepare():
    import soundfile as sf
    out=ROOT/'output/validation/p17-m04-r2'/uuid4().hex
    inputs=out/'inputs';inputs.mkdir(parents=True)
    saved=out/'saved';saved.mkdir()
    egg=CORPUS/'EGG测试/3.wav'
    candidates=sorted(CORPUS.glob('已标注音频/**/*.wav'))
    short=next(p for p in candidates if p.with_suffix('.TextGrid').exists() and p.with_suffix('.xlsx').exists() and .5<sf.info(str(p)).duration<10)
    selected=[(short,short.name),(egg,'EGG-real.wav'),(egg,'EGG-second.wav')]
    manifest=[]
    for source,name in selected:
        target=inputs/name;shutil.copyfile(source,target)
        info=sf.info(str(source));sha=hashlib.sha256(source.read_bytes()).hexdigest()
        manifest.append(dict(source=str(source.relative_to(CORPUS)),copy=name,sha256=sha,sample_rate=info.samplerate,channels=info.channels,frames=info.frames,duration=info.duration))
        grid=source.with_suffix('.TextGrid')
        if grid.exists():shutil.copyfile(grid,target.with_suffix('.TextGrid'))
        for ext in ('.xlsx','.ptb.sqlite','.ptb.sqlite3'):
            table=source.with_suffix(ext)
            if table.exists():shutil.copyfile(table,target.with_suffix(ext))
    (out/'inputs.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    database,cache=prepare_workspace(out/'workspace',ROOT/'backend/migrations')
    os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
    os.environ['PTB_M05_PYTHON']=str(ROOT/'.venv/m05/Scripts/python.exe')
    return out,inputs,saved,database,cache,manifest

def main():
    out,inputs,saved,db,cache,manifest=prepare();provider=FileProvider()
    try:
        with LocalService(db,local_files_root=cache,reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe') as service:
            bridge=TaskBridge(provider,service)
            grants={role:provider.choose(role,lambda p=p:str(p)) for role,p in [('input',inputs),('output',saved),('association',saved)]}
            print(json.dumps(dict(ready=True,out=str(out),short=manifest[0]['copy']),ensure_ascii=False),flush=True)
            for line in sys.stdin:
                request={}
                try:
                    request=json.loads(line);rpc_id=request.pop('rpc_id');op=request['op']
                    if op=='shutdown':break
                    if op=='choose':value=grants[request['purpose']]
                    elif op=='files':value=(provider.scan if request.get('recursive') else provider.list)(request.get('directory') or grants['input']['id'])
                    elif op in ('read','textgrid','spectrogram'):
                        raw,digest=provider.read(request['id'])
                        if op=='read':value={'base64':base64.b64encode(raw).decode(),'sha256':digest}
                        elif op=='textgrid':value={'sha256':digest,'tiers':[asdict(t) for t in parse_textgrid(raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'))]}
                        else:value=service.preview(raw,request['view'])
                    else:value=bridge.invoke(request)
                    print(json.dumps(dict(id=rpc_id,value=value),ensure_ascii=False),flush=True)
                except Exception as exc:
                    print(json.dumps(dict(id=locals().get('rpc_id'),error=str(exc)),ensure_ascii=False),flush=True)
    finally:
        provider.close()
        unchanged=all(hashlib.sha256((CORPUS/m['source']).read_bytes()).hexdigest()==m['sha256'] for m in manifest)
        (out/'originals-unchanged.json').write_text(json.dumps(dict(unchanged=unchanged)),encoding='utf8')

if __name__=='__main__':main()
