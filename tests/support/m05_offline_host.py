"""Actual local API/worker/export bridge in owned scratch, without devices."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from workbench_source import configure
configure()
from verification_artifacts import verification_run
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.task_bridge import TaskBridge
from ptb_desktop.local_service import LocalService
from ptb_worker.local_acoustic_files import initialize_local_files

def main():
    with verification_run('m05-offline','chrome','Public NASA astronaut image; two 6-frame 10fps FFV1 videos with 1s/8000Hz sine audio; actual analysis and export; no camera/microphone') as (out,scratch):
        db=scratch/'jobs.sqlite3'
        with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
            source.backup(target)
            # This fresh test copy owns its rows; the development DB is read-only.
            target.execute('DELETE FROM events');target.execute('DELETE FROM jobs');target.commit()
        inputs=scratch/'input';inputs.mkdir();saved=scratch/'saved';saved.mkdir();cache=scratch/'cache';cache.mkdir();initialize_local_files(cache)
        def probe(*args):
            subprocess.run([os.environ['PTB_M05_PYTHON'],'-X','utf8',str(ROOT/'tests/support/m05_offline_media.py'),*map(str,args)],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        probe('generate',inputs)
        provider=FileProvider();destination=provider.choose('output',lambda:str(saved))
        with LocalService(db,local_files_root=cache) as service:
            bridge=TaskBridge(provider,service)
            print(json.dumps(dict(ready=True,out=str(out),scratch=str(scratch),inputs=str(inputs),saved=str(saved))),flush=True)
            for line in sys.stdin:
                request=json.loads(line);body=request['body']
                if body.get('op')=='shutdown':break
                try:
                    if request['channel']=='task':value=bridge.invoke(body)
                    elif body['op']=='hello':value=dict(kind='desktop',session=provider.session,api_version='1.1.0',tasks=True)
                    elif body['op']=='fonts':value=['Arial','Microsoft YaHei','Doulos SIL']
                    elif body['op']=='choose':value=destination
                    else:raise ValueError('Unsupported test transport')
                    response=dict(id=request['id'],ok=True,value=value)
                except Exception as error:
                    response=dict(id=request['id'],ok=False,error=str(error))
                    with (out/'transport-errors.jsonl').open('a',encoding='utf8') as log:log.write(json.dumps(dict(op=body.get('op'),error=repr(error)),ensure_ascii=False)+'\n')
                print(json.dumps(response,ensure_ascii=False),flush=True)
        if (out/'report.json').is_file() and json.loads((out/'report.json').read_text('utf8')).get('success'):probe('inspect',saved,out)

if __name__=='__main__':main()
