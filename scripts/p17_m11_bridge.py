"""Test transport only: actual FileProvider/TaskBridge/LocalService and worker."""
import base64
import json
from pathlib import Path
import sys
import os
import shutil
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.task_bridge import TaskBridge
from ptb_desktop.local_service import LocalService
from uuid import uuid4
from ptb_worker.local_workspace import prepare_workspace
ROOT=Path(__file__).resolve().parents[1]
def setup():
    base=(ROOT/'output/validation/p17/M11-ui').resolve();out=Path(os.environ['P17_REUSE_M11']).resolve() if os.environ.get('P17_REUSE_M11') else base/uuid4().hex
    if not out.is_relative_to(base):raise ValueError('Test output must stay in P17/M11-ui')
    out.mkdir(parents=True,exist_ok=True)
    components=out/'components';components.mkdir(exist_ok=True);shutil.copyfile(ROOT/'output/validation/m11/qt-642b638d008c497abb576e5db63dcbe7/components/registry.json',components/'registry.json');os.environ['PTB_M11_COMPONENT_ROOT']=str(components)
    db,cache=prepare_workspace(out/'workspace',ROOT/'backend/migrations')
    return out,db,cache


def main():
    out,db,cache=setup();inputs=out/'input';inputs.mkdir(exist_ok=True);saved=out/'saved';saved.mkdir(exist_ok=True)
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
                    elif op=='m11_pick':
                        purpose=body['purpose'];registry=json.loads((out/'components/registry.json').read_text('utf8'));chosen={'runtime':registry['runtimes'][0]['path'],'model':registry['models'][0]['model'],'dictionary':registry['models'][0]['dictionary']}.get(purpose)
                        if purpose=='corpus':
                            lab=Path('C:/Users/13680/Desktop/project/音频数据/creak/老年组 15人/1.凌静梅/00141低_梯_题.lab')
                            for source in [lab,lab.with_suffix('.wav')]:
                                if not (inputs/source.name).exists():shutil.copyfile(source,inputs/source.name)
                            chosen=inputs
                        value=bridge.m11.grant(purpose,chosen) if chosen else None
                    elif op=='choose':value=destination if body['purpose']=='output' else directory
                    elif op=='list':value=provider.list(body.get('id') or directory['id'])
                    elif op=='read':
                        raw,sha=provider.read(body['id']);value=dict(base64=base64.b64encode(raw).decode(),sha256=sha)
                    else:raise ValueError('Unsupported test transport')
                response=dict(id=request['id'],ok=True,value=value)
            except Exception as exc:response=dict(id=request['id'],ok=False,error=str(exc))
            print(json.dumps(response,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
