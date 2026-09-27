"""Existing unified local HTTP host, isolated copy of existing schema, synthetic data."""
import json
import sqlite3
import time
from pathlib import Path
from uuid import uuid4
from ptb_desktop.local_service import LocalService
from ptb_worker.local_acoustic_files import initialize_local_files
from ptb_worker.store import LOCAL_PROJECT

ROOT=Path(__file__).resolve().parents[1]


def setup():
    out=ROOT/'output/validation/m14/wiring'/uuid4().hex;out.mkdir(parents=True)
    db=out/'jobs.sqlite3';template=ROOT/'output/validation/p06/local-state.sqlite3'
    with sqlite3.connect(template.as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone();source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    return out,db,cache


def wait(service,job):
    until=time.monotonic()+90
    while time.monotonic()<until:
        value=service.get('/api/v1/jobs/'+job['id'])
        if value['state'] not in ('queued','running','cancel_requested'):return value
        time.sleep(.1)
    raise AssertionError('m14 wait timed out')


def read(service,file):
    return b''.join(service.binary(f'/api/v1/jobs/local-results/{file["id"]}?offset={i}&size={min(1048576,file["size_bytes"]-i)}') for i in range(0,file['size_bytes'],1048576))


def main():
    from hashlib import sha256
    out,db,cache=setup();checks=[];report=dict(success=False,checks=checks,scope='Windows production local HTTP/persistent worker, synthetic only; no DDL')
    print(out,flush=True)
    try:
        with LocalService(db,local_files_root=cache) as service:
            for ext in ('xlsx','xls','csv','txt','tsv'):
                name='public.'+ext;ref=service.import_input((ROOT/'tests/fixtures/m14'/name).read_bytes(),name,'table')
                body=dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,table=ref,config=dict(action='preview',skip_first_row=True,consonant_only_as_zero_initial=True))
                job=service.request('/api/v1/jobs/m14/create','POST',body)
                assert service.request('/api/v1/jobs/m14/create','POST',body)['id']==job['id']
                done=wait(service,job);assert done['state']=='succeeded',done
                preview=json.loads(read(service,done['result_manifest']['files'][0]));checks.append(ext+' real persistent preview')
            body['idempotency_key']=uuid4().hex;body['config'].update(action='export',settings=preview['config'],font=dict(schema_version='font/1',zh='Microsoft YaHei',latin='Segoe UI',ipa='Doulos SIL',size_px=14))
            done=wait(service,service.request('/api/v1/jobs/m14/create','POST',body));assert done['state']=='succeeded',done
            assert len(done['result_manifest']['files'])==3
            for f in done['result_manifest']['files']:
                raw=read(service,f);assert sha256(raw).hexdigest()==f['sha256'];(out/f['name']).write_bytes(raw)
            checks.append('three-file atomic manifest and byte/hash readback')
            ref=service.import_input(b'bad','bad.xlsx','table');body.update(table=ref,idempotency_key=uuid4().hex);body['config']['action']='preview'
            done=wait(service,service.request('/api/v1/jobs/m14/create','POST',body));assert done['state']=='failed' and done['error_code']=='m14_decode_failed',done
            checks.append('corrupt input failure preserves completed outputs')
            body['idempotency_key']=uuid4().hex;job=service.request('/api/v1/jobs/m14/create','POST',body);service.request('/api/v1/jobs/'+job['id']+'/cancel','POST');done=wait(service,job);assert done['state']=='cancelled',done;checks.append('durable cancellation')
            report['success']=True
    finally:(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
