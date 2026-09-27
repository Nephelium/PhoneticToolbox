"""Real local service + persistent SQLite copy + bounded Praat process. No DDL."""
import base64
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4
from ptb_desktop.local_service import LocalService
from ptb_worker.local_acoustic_files import initialize_local_files
from ptb_worker.store import LOCAL_PROJECT

ROOT=Path(__file__).resolve().parents[1]


def wait(service,job):
    deadline=time.monotonic()+150
    while time.monotonic()<deadline:
        result=service.get('/api/v1/jobs/'+job['id'])
        if result['state'] not in ('queued','running','cancel_requested'):return result
        time.sleep(.1)
    raise AssertionError('Job did not settle')


def setup():
    out=ROOT/'output/validation/m08-wiring'/uuid4().hex;out.mkdir(parents=True)
    db=out/'jobs.sqlite3'; template=ROOT/'output/validation/p06/local-state.sqlite3'
    with sqlite3.connect(template.as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
        source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    return out,db,cache


def main():
    out,db,cache=setup();print(out,flush=True)
    report={'checks':[],'success':False}
    try:
        with LocalService(db,local_files_root=cache) as service:
            refs={}
            for extension in ('wav','mp3','flac'):
                raw=(ROOT/'output/validation/m08-wiring'/('input.'+extension)).read_bytes()
                ref=service.import_input(raw,'合成 ɑ̃˥.'+extension,'audio');refs[extension]=ref
                job=service.request('/api/v1/jobs/m08/create','POST',dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,audio=ref,config={'action':'preview'}))
                done=wait(service,job);assert done['state']=='succeeded',done
                report['checks'].append(extension+' actual decode / durable preview')
            ref=refs['wav']
            body=dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,audio=ref,config={'action':'transform','speed':.8,'pitch_ratio':1.2,'pitch_hz':20})
            job=service.request('/api/v1/jobs/m08/create','POST',body)
            assert service.request('/api/v1/jobs/m08/create','POST',body)['id']==job['id']
            done=wait(service,job);assert done['state']=='succeeded',done
            listing=service.get('/api/v1/jobs/m08/list/'+LOCAL_PROJECT)
            result=next(j for j in listing if j['id']==job['id'])['results'][0]
            report['checks'].append('persistent transform and idempotency')
            scope=dict(project_id=LOCAL_PROJECT,source=ref,ids=[result['id']])
            saved=[]
            for _ in range(2):
                copy=service.request('/api/v1/jobs/m08/save','POST',scope);done=wait(service,copy)
                assert done['state']=='succeeded',done
                saved.extend(next(j for j in service.get('/api/v1/jobs/m08/list/'+LOCAL_PROJECT) if j['id']==copy['id'])['results'])
            assert saved[0]['name'].endswith('_1.wav') and saved[1]['name'].endswith('_2.wav'),saved
            assert saved[0]['sha256']==saved[1]['sha256']==result['sha256']
            report['checks'].append('atomic numbered copies preserve PCM hashes')
            scope['ids']=[r['id'] for r in saved]
            service.request('/api/v1/jobs/m08/rename','POST',dict(scope,names=['重命名_1.wav','重命名_2.wav']))
            history=service.request('/api/v1/jobs/m08/history','POST',dict(project_id=LOCAL_PROJECT,source=ref))
            assert {'重命名_1.wav','重命名_2.wav'} <= {r['name'] for r in history}
            removed=service.request('/api/v1/jobs/m08/remove','POST',scope)
            assert removed=={'removed':scope['ids'],'failed':[]},removed
            report['checks'].append('explicit result ID rename/delete and zero-based history')
        with LocalService(db,local_files_root=cache) as service:
            assert service.get('/api/v1/jobs/'+job['id'])['state']=='succeeded'
        report['checks'].append('host restart preserves original task')
        state=json.loads((cache/'.ptb-local.json').read_text('utf8'))
        assert not [a for a in state['assets'].values() if a['reserved_bytes'] or a['kind']=='temporary' and a['state']!='deleted']
        report['checks'].append('no temporary files or reservations remain')
        report['success']=True
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
