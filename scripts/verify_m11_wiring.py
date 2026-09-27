"""Actual formal local HTTP/SQLite worker with optional MFA and public inputs.

Uses an isolated copy of the existing empty template. No existing database DDL.
"""
import argparse
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
from ptb_worker.mfa.probe import generate,register

ROOT=Path(__file__).resolve().parents[1]


def wait(service,job):
    end=time.monotonic()+240
    while time.monotonic()<end:
        job=service.get('/api/v1/jobs/'+job['id'])
        if job['state'] not in ('queued','running','cancel_requested'):return job
        time.sleep(.2)
    raise AssertionError('m11 formal task timed out')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',required=True)
    parser.add_argument('--model',required=True)
    a=parser.parse_args()
    out=ROOT/'output/validation/m11'/('wiring-'+uuid4().hex);out.mkdir(parents=True)
    print(str(out),flush=True)
    os.environ['PTB_M11_COMPONENT_ROOT']=str(out/'components')
    generate(out/'public')
    dictionary=out/'probe.dict';dictionary.write_text('a\ta˥˥\n',encoding='utf8')
    checks=[];report=dict(success=False,checks=checks,scope='actual Windows local HTTP + persistent MFA; public synthetic input')
    try:
        registered=register(a.runtime,a.model,dictionary)
        checks.append('optional environment actual small-task qualification')
        db=out/'jobs.sqlite3'
        template=ROOT/'output/validation/p06/local-state.sqlite3'
        with sqlite3.connect(template.as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
            assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
            source.backup(target)
        cache=out/'cache';cache.mkdir();initialize_local_files(cache)
        with LocalService(db,local_files_root=cache) as service:
            audio=service.import_input((out/'public/probe.wav').read_bytes(),'中文 空格.wav','audio')
            text=service.import_input((out/'public/probe.lab').read_bytes(),'中文 空格.lab','transcript')
            body=dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,runtime_id=registered['runtime_id'],model_id=registered['model_id'],
                      corpus=[dict(name='中文 空格.wav',audio=audio,transcript=text)],config=dict(beam=10,retry_beam=40))
            job=service.request('/api/v1/jobs/m11/create','POST',body)
            assert service.request('/api/v1/jobs/m11/create','POST',body)['id']==job['id']
            done=wait(service,job);assert done['state']=='succeeded',done
            for f in done['result_manifest']['files']:
                raw=b''.join(service.binary(f'/api/v1/jobs/local-results/{f["id"]}?offset={i}&size={min(1048576,f["size_bytes"]-i)}') for i in range(0,f['size_bytes'],1048576))
                assert hashlib.sha256(raw).hexdigest()==f['sha256']
                (out/f['name']).write_bytes(raw)
            checks.append('formal HTTP create -> worker -> real MFA -> atomic results -> byte/hash readback')
            report['job_id']=done['id']
            report['provenance']=json.loads((out/'m11-provenance.json').read_text(encoding='utf8'))
            body['idempotency_key']=uuid4().hex
            second=service.request('/api/v1/jobs/m11/create','POST',body)
            end=time.monotonic()+30
            while time.monotonic()<end:
                status=service.get('/api/v1/jobs/'+second['id'])
                if status['state']=='running':break
                time.sleep(.1)
            time.sleep(2)
            service.request('/api/v1/jobs/'+second['id']+'/cancel','POST')
            cancelled=wait(service,second)
            assert cancelled['state']=='cancelled',cancelled
            checks.append('real running MFA cancellation through formal job API')
        with LocalService(db,local_files_root=cache) as service:
            restored=service.get('/api/v1/jobs/'+report['job_id'])
            assert restored['state']=='succeeded' and restored['result_manifest']==done['result_manifest']
            checks.append('formal host restart preserves job and complete manifest')
        report['success']=True
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(success=report['success'],checks=checks,root=str(out)),ensure_ascii=False))


if __name__=='__main__':main()
