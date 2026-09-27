"""Production LocalService with an isolated COPY of an existing schema, no DDL."""
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4
from ptb_desktop.local_service import LocalService
from ptb_worker.local_acoustic_files import initialize_local_files
from ptb_worker.store import LOCAL_PROJECT
from phonetic_core.synthesis.klatt.api import defaults,export_parameters

ROOT=Path(__file__).resolve().parents[1]


def setup():
    out=ROOT/'output/validation/m06/host'/uuid4().hex;out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    return out,db,cache


def wait(service,job):
    deadline=time.monotonic()+150
    while time.monotonic()<deadline:
        value=service.get('/api/v1/jobs/'+job['id'])
        if value['state'] not in ('queued','running','cancel_requested'):return value
        time.sleep(.1)
    raise AssertionError('M06 host timeout')


def read(service,f):
    return b''.join(service.binary(f'/api/v1/jobs/local-results/{f["id"]}?offset={i}&size={min(1048576,f["size_bytes"]-i)}') for i in range(0,f['size_bytes'],1048576))


def main():
    from scipy.io import wavfile
    out,db,cache=setup();report=dict(success=False,checks=[]);print(out,flush=True)
    try:
        with LocalService(db,local_files_root=cache) as service:
            assert 'M06' in service.get('/api/v1/capabilities')['algorithms']
            c=defaults();c['sequence']='a i';c['duration']=.6
            for curve in c['curves'].values():curve['points'][-1][0]=.6
            for action in ('generate','synthesize','extract'):
                body=dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,action=action,parameters=service.import_input(export_parameters(c).encode(),'m06-parameters.csv','table'))
                if action=='extract':body['audio']=service.import_input((ROOT/'tests/fixtures/m06/source.wav').read_bytes(),'source.wav','audio')
                created=service.request('/api/v1/jobs/m06/create','POST',body)
                assert service.request('/api/v1/jobs/m06/create','POST',body)['id']==created['id']
                job=wait(service,created);assert job['state']=='succeeded',job
                data={f['name']:read(service,f) for f in job['result_manifest']['files']}
                for f in job['result_manifest']['files']:assert hashlib.sha256(data[f['name']]).hexdigest()==f['sha256']
                c=json.loads(data['m06.ptb.json'])['config']
                if action=='synthesize':
                    rate,samples=wavfile.read(io.BytesIO(data['synthesis.wav']));assert rate==16000 and len(samples)==9600
                    for name,raw in data.items():(out/name).write_bytes(raw)
                report['checks'].append(action+' production HTTP + persistent worker + artifact hashes')
            body=dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,action='generate',parameters=service.import_input(export_parameters(dict(c,sequence='?')).encode(),'m06-parameters.csv','table'))
            failed=wait(service,service.request('/api/v1/jobs/m06/create','POST',body));assert failed['state']=='failed'
            report['checks'].append('failed generation keeps prior published audio')
            report['success']=True
    finally:(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report))


if __name__=='__main__':main()
