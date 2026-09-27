"""M07 real LocalService HTTP on a task-owned existing-schema copy."""
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
ROOT=Path(__file__).resolve().parents[1]
def setup():
    out=ROOT/'output/validation/m07/host'/uuid4().hex;out.mkdir(parents=True);db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache);return out,db,cache
def wait(service,job):
    deadline=time.monotonic()+150
    while time.monotonic()<deadline:
        value=service.get('/api/v1/jobs/'+job['id'])
        if value['state'] not in ('queued','running','cancel_requested'):return value
        time.sleep(.1)
    raise AssertionError('M07 host timeout')
def read(service,f):return b''.join(service.binary(f'/api/v1/jobs/local-results/{f["id"]}?offset={i}&size={min(1048576,f["size_bytes"]-i)}') for i in range(0,f['size_bytes'],1048576))
def main():
    from scipy.io import wavfile
    import numpy as np
    out,db,cache=setup();report=dict(success=False,checks=[]);print(out,flush=True)
    try:
        with LocalService(db,local_files_root=cache) as service:
            assert 'M07' in service.get('/api/v1/capabilities')['algorithms']
            refs=[service.import_input((ROOT/f'output/validation/m07/baseline/round1/input{i}.wav').read_bytes(),f'input{i}.wav','audio') for i in (0,1)]
            def run(action,**kw):
                body=dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,action=action,source=refs[0],target=refs[1],**kw)
                created=service.request('/api/v1/jobs/m07/create','POST',body)
                assert service.request('/api/v1/jobs/m07/create','POST',body)['id']==created['id']
                job=wait(service,created);assert job['state']=='succeeded',job
                data={f['name']:read(service,f) for f in job['result_manifest']['files']}
                for f in job['result_manifest']['files']:assert hashlib.sha256(data[f['name']]).hexdigest()==f['sha256']
                return job,data
            job,data=run('analyze');meta=json.loads(data['m07.ptb.json']);report['checks'].append('production HTTP analysis, idempotency, artifact hashes')
            applied,_=run('apply',analysis_job_id=job['id'],controls=meta['controls'])
            report['checks'].append('production explicit apply with immutable analysis snapshot')
            for reverse in (False,True):
                for kind in (2,1,3):
                    generated,raw=run('generate',analysis_job_id=applied['id'],continuum_type=kind,reverse_direction=reverse,generation={'step_count':3})
                    parts=[wavfile.read(io.BytesIO(raw[f'step{i:02d}.wav']))[1] for i in range(1,4)]
                    rate,combined=wavfile.read(io.BytesIO(raw['combined_steps.wav']));assert rate==11025
                    np.testing.assert_array_equal(combined,np.concatenate(parts));report['checks'].append(f'{reverse}/{kind} complete WAV readback')
                    folder=out/generated['id'];folder.mkdir()
                    for name,value in raw.items():(folder/name).write_bytes(value)
            report['success']=True
    finally:(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report))
if __name__=='__main__':main()
