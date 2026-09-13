"""M04-D owned stdin RPC to real local service and scientific child; no DDL."""
import base64
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
from phonetic_core.textgrid import parse_textgrid
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.local_service import LocalService
from ptb_desktop.task_bridge import TaskBridge
from ptb_worker.local_acoustic_files import initialize_local_files

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'output/validation/m04-ui'/uuid4().hex;out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
        source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    t=np.arange(96000)/48000;audio=.3*np.sin(2*np.pi*150*t)+.1*np.sin(2*np.pi*800*t)
    wavfile.write(inputs/'LPC ɑ̃˥.wav',48000,np.column_stack([audio,audio*.5]))
    wavfile.write(inputs/'second.wav',48000,audio*.5)
    wavfile.write(inputs/'silent.wav',48000,np.zeros(96000))
    grid='File type = "ooTextFile"\nObject class = "TextGrid"\n\n0\n2\n<exists>\n2\n"IntervalTier"\n"phones"\n0\n2\n2\n0\n1\n"ɑ̃˥"\n1\n2\n"末"\n"IntervalTier"\n"words"\n0\n2\n1\n0\n2\n"音节"\n'
    (inputs/'LPC ɑ̃˥.TextGrid').write_text(grid,encoding='utf-8')
    os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
    provider=FileProvider()
    try:
        with LocalService(db,local_files_root=cache) as service:
            bridge=TaskBridge(provider,service)
            grants={role:provider.choose(role,lambda p=p:str(p)) for role,p in [('input',inputs),('output',saved)]}
            print(json.dumps({'ready':True,'out':str(out)},ensure_ascii=False),flush=True)
            for line in sys.stdin:
                request={}
                try:
                    request=json.loads(line);op=request['op']
                    if op=='shutdown':break
                    if op=='choose':value=grants[request['purpose']]
                    elif op=='list':value=provider.list(grants['input']['id'])
                    elif op in ('read','textgrid'):
                        raw,digest=provider.read(request['id'])
                        value={'base64':base64.b64encode(raw).decode(),'sha256':digest} if op=='read' else {'sha256':digest,'tiers':[asdict(t) for t in parse_textgrid(raw.decode('utf-8'))]}
                    else:value=bridge.invoke(request)
                    print(json.dumps({'id':request['rpc_id'],'value':value},ensure_ascii=False),flush=True)
                except Exception as e:print(json.dumps({'id':request.get('rpc_id'),'error':str(e)},ensure_ascii=False),flush=True)
    finally:provider.close()

if __name__=='__main__':main()
