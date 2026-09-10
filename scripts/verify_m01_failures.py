"""M01-G actual offline science failure publication, cleanup and recovery.

Uses only already-approved SQLite tables plus a new owned synthetic cache. No DDL.
"""
import io
import json
from pathlib import Path
import time
from uuid import uuid4
import wave
from baseline_support import RECIPES,create_fixture
from ptb_desktop.local_service import LocalService
from ptb_worker.local_acoustic_files import initialize_local_files

ROOT=Path(__file__).resolve().parents[1]
PROJECT='00000000-0000-4000-8000-000000000001'


def main():
    out=ROOT/'output/validation/m01'/('failures-'+uuid4().hex);out.mkdir()
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    raw=create_fixture(out,RECIPES[0]).read_bytes()
    def wav(frames):
        result=io.BytesIO()
        with wave.open(result,'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(8000);f.writeframes(b'\x00\x00'*frames)
        return result.getvalue()
    cases=[('bad-wav',{'audio':b'private invalid wave text'},'invalid_audio'),
        ('bad-grid',{'audio':raw,'textgrid':b'private invalid annotation'},'invalid_textgrid'),
        ('bad-lip',{'audio':raw,'lip':b'{"private":true}'},'invalid_lip'),
        ('empty',{'audio':wav(0)},'no_parameter_frames'),
        ('long',{'audio':wav(2_000_001)},'analysis_sample_limit'),
        ('good',{'audio':raw},None)]
    options=dict(local_files_root=cache,reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe')
    rows=[];batch_ids=[]
    with LocalService(ROOT/'output/validation/p06/local-state.sqlite3',**options) as service:
        try:
            for name,blobs,code in cases:
                refs={role:service.import_input(data,name+{'audio':'.wav','textgrid':'.TextGrid','lip':'.lip.json'}[role],role) for role,data in blobs.items()}
                batch=service.request('/api/v1/jobs/batches/create','POST',dict(project_id=PROJECT,operation='acoustic_analysis',
                    inputs=[refs],config=dict(selection=dict(keys=['pF0']),backend_policy=dict(reaper='native_required')),idempotency_key=uuid4().hex))
                batch_ids.append(batch['id']);deadline=time.monotonic()+75
                while not batch['summary']['closed'] and time.monotonic()<deadline:
                    time.sleep(.15);batch=service.get('/api/v1/jobs/batches/'+batch['id'])
                assert batch['summary']['closed'],name+' timeout'
                item=batch['summary']['items'][0];job=service.get('/api/v1/jobs/'+item['job_id'])
                assert job['error_code']==item['error_code']==code,(name,job['error_code'])
                assert job['state']==('failed' if code else 'succeeded')
                assert bool(job['result_manifest'])==(code is None)
                metadata=json.loads((cache/'.ptb-local.json').read_text('utf-8'))
                assert not any(a['kind']=='temporary' and a['state']!='deleted' for a in metadata['assets'].values())
                rows.append(dict(case=name,batch=batch['id'],error_code=code,state=job['state'],temporary_remaining=0))
        finally:
            for key in batch_ids:service.request('/api/v1/jobs/batches/'+key+'/cancel','POST')
    with LocalService(ROOT/'output/validation/p06/local-state.sqlite3',**options) as service:
        for row in rows:
            batch=service.get('/api/v1/jobs/batches/'+row['batch'])
            assert batch['summary']['closed'] and batch['summary']['items'][0]['error_code']==row['error_code']
    report=dict(task='M01-G',cases=rows,restart_preserves_reasons=True,scope='actual local service and owned scientific child; synthetic only')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(str((out/'report.json').relative_to(ROOT)))


if __name__=='__main__':main()
