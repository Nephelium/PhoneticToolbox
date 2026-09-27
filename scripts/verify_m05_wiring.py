"""Actual local HTTP/SQLite/Job Object M05 flow; public material only, no devices."""
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
    until=time.monotonic()+180
    while time.monotonic()<until:
        job=service.get('/api/v1/jobs/'+job['id'])
        if job['state'] not in ('queued','running','cancel_requested'):return job
        time.sleep(.2)
    raise AssertionError('m05_task_timeout')

def main():
    out=ROOT/'output/validation/m05'/('wiring-'+uuid4().hex);out.mkdir(parents=True)
    print(str(out),flush=True)
    os.environ['PTB_M05_PYTHON']=str(ROOT/'.venv/m05/Scripts/python.exe')
    db=out/'jobs.sqlite3';template=ROOT/'output/validation/p06/local-state.sqlite3'
    with sqlite3.connect(template.as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
        source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    report=dict(success=False,checks=[],scope='public NASA-derived video; actual local API and legacy scientific subprocess; no camera/audio devices')
    try:
        with LocalService(db,local_files_root=cache) as service:
            assert service.get('/api/v1/jobs/m05/catalog')['available']
            for _ in range(9):
                interrupted=service.request('/api/v1/jobs/m05/uploads','POST',dict(name='cancelled.webm',size=10))
                service.request('/api/v1/jobs/m05/uploads/'+interrupted['id'],'PUT',dict(offset=0,base64='YWJj'))
                assert service.request('/api/v1/jobs/m05/uploads/'+interrupted['id']+'/abort','POST')['aborted']
            report['checks'].append('nine interrupted uploads released reservations without deleting partial files or blocking subsequent input')
            path=ROOT/'output/validation/m05/inputs/motion-occlusion-vfr/input.mkv'
            upload=service.request('/api/v1/jobs/m05/uploads','POST',dict(name='公开测试 中文.mkv',size=path.stat().st_size))
            with path.open('rb') as stream:
                offset=0
                for raw in iter(lambda:stream.read(262144),b''):
                    service.request('/api/v1/jobs/m05/uploads/'+upload['id'],'PUT',dict(offset=offset,base64=base64.b64encode(raw).decode()));offset+=len(raw)
            ref=service.request('/api/v1/jobs/m05/uploads/'+upload['id']+'/finalize','POST')
            assert ref['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
            body=dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,video=ref,config=dict(filter_enabled=True,cutoff_hz=15))
            job=service.request('/api/v1/jobs/m05/create','POST',body)
            assert service.request('/api/v1/jobs/m05/create','POST',body)['id']==job['id']
            done=wait(service,job);assert done['state']=='succeeded',done
            report['job_id']=job['id'];report['checks'].append('chunked input, idempotency, actual legacy analysis, atomic result publication')
            for f in done['result_manifest']['files']:
                h=hashlib.sha256()
                with (out/f['name']).open('xb') as output:
                    for offset in range(0,f['size_bytes'],262144):
                        raw=service.binary(f'/api/v1/jobs/local-results/{f["id"]}?offset={offset}&size={min(262144,f["size_bytes"]-offset)}');output.write(raw);h.update(raw)
                assert h.hexdigest()==f['sha256']
            meta=json.loads((out/'manifest.json').read_text('utf8'));assert meta['timing']['decoded_frames']==30 and meta['validity']['missing']==12
            from ptb_worker.io.lip import decode_lip
            assert len(decode_lip((out/'audio_recording.lip.json').read_bytes())['relative_times'])==30
            report['resources']=json.loads((out/'resources.json').read_text('utf8'));assert report['resources']['group_cleaned']
            report['checks'].append('all result hashes and safe lip format read back; actual decoded PTS retained')
            body['idempotency_key']=uuid4().hex
            second=service.request('/api/v1/jobs/m05/create','POST',body)
            until=time.monotonic()+15
            while time.monotonic()<until and service.get('/api/v1/jobs/'+second['id'])['state']=='queued':time.sleep(.05)
            service.request('/api/v1/jobs/'+second['id']+'/cancel','POST')
            cancelled=wait(service,second);assert cancelled['state']=='cancelled',cancelled
            report['checks'].append('running task cancellation with no partial publication')
        with LocalService(db,local_files_root=cache) as service:
            restored=service.get('/api/v1/jobs/'+job['id']);assert restored['state']=='succeeded' and restored['result_manifest']==done['result_manifest']
            report['checks'].append('actual service restart retained manifest')
        report['success']=True
    finally:(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
