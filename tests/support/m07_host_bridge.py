"""Test transport only: actual FileProvider/TaskBridge/LocalService and worker."""
import base64
import json
from pathlib import Path
import sys
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.task_bridge import TaskBridge
from ptb_desktop.local_service import LocalService
from verify_m07_wiring import setup,ROOT


def main():
    out,db,cache=setup();inputs=out/'input';inputs.mkdir();saved=out/'saved';saved.mkdir()
    for i in (0,1):
        (inputs/f'input{i}.wav').write_bytes((ROOT/f'output/validation/m07/baseline/round1/input{i}.wav').read_bytes())
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
                    elif op=='fonts':value=['Arial','Microsoft YaHei','Doulos SIL']
                    elif op=='choose':value=destination if body['purpose']=='output' else directory
                    elif op=='list':value=provider.list(body.get('id') or directory['id'])
                    elif op=='read':
                        raw,sha=provider.read(body['id']);value=dict(base64=base64.b64encode(raw).decode(),sha256=sha)
                    else:raise ValueError('Unsupported test transport')
                response=dict(id=request['id'],ok=True,value=value)
            except Exception as exc:response=dict(id=request['id'],ok=False,error=str(exc))
            print(json.dumps(response,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
