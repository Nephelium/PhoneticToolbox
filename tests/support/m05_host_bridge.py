"""Test transport only: actual FileProvider/TaskBridge/LocalService and worker."""
import base64
import json
from pathlib import Path
import sys
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.task_bridge import TaskBridge
from ptb_desktop.local_service import LocalService
import sqlite3
import os
from uuid import uuid4
from ptb_worker.local_acoustic_files import initialize_local_files
ROOT=Path(__file__).resolve().parents[2]
def setup():
    out=ROOT/'output/validation/m05'/('host-'+uuid4().hex);out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    os.environ['PTB_M05_PYTHON']=str(ROOT/'.venv/m05/Scripts/python.exe')
    return out,db,cache


def main():
    out,db,cache=setup();inputs=out/'input';inputs.mkdir();saved=out/'saved';saved.mkdir()

    provider=FileProvider();directory=provider.choose('input',lambda:str(inputs));destination=provider.choose('output',lambda:str(saved))
    with LocalService(db,local_files_root=cache) as service:
        bridge=TaskBridge(provider,service)
        print(json.dumps(dict(ready=True,out=str(out))),flush=True)
        for line in sys.stdin:
            request=json.loads(line);body=request['body']
            try:
                if request['channel']=='task':value=bridge.invoke(body)
                else:
                    op=body['op']
                    if op=='hello':value=dict(kind='desktop',session=provider.session,api_version='1.1.0',tasks=True)
                    elif op=='m05_media':value=True  # test-only transport arm; synthetic stream below
                    elif op=='fonts':value=['Arial','Microsoft YaHei','Doulos SIL']
                    elif op=='choose':value=destination if body['purpose']=='output' else directory
                    elif op=='list':value=provider.list(body.get('id') or directory['id'])
                    elif op=='read':
                        raw,sha=provider.read(body['id']);value=dict(base64=base64.b64encode(raw).decode(),sha256=sha)
                    else:raise ValueError('Unsupported test transport')
                response=dict(id=request['id'],ok=True,value=value)
            except Exception as exc:
                response=dict(id=request['id'],ok=False,error=str(exc))
                with (out/'transport-errors.jsonl').open('a',encoding='utf8') as log:log.write(json.dumps(dict(op=body.get('op'),error=str(exc)),ensure_ascii=False)+'\n')
            print(json.dumps(response,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
