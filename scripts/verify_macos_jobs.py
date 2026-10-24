"""Actual local API admission, durable result publication and cancellation."""
import io,json,secrets,threading
from pathlib import Path
from uuid import uuid4

def main(output,fixtures):
    import numpy as np,soundfile as sf
    from fastapi.testclient import TestClient
    from ptb_worker.native.darwin import bundle_root
    from ptb_worker.local_workspace import prepare_workspace
    from ptb_worker.store import SQLiteJobStore,LOCAL_PROJECT
    from ptb_worker.local_acoustic_files import LocalAcousticFiles
    from ptb_worker.acoustic_batches import AcousticBatches
    from ptb_api.main import create_app
    from ptb_worker.acoustic_executor import execute_acoustic_claim
    from ptb_worker.m14_executor import execute_claim as execute_m14
    output=Path(output);output.mkdir(parents=True,exist_ok=False);root=bundle_root()
    db,cache=prepare_workspace(output/'scratch'/'workspace',root/'backend/migrations')
    store=SQLiteJobStore(db);files=LocalAcousticFiles(store,cache,reaper_binary=root/'resources/research/reaper');AcousticBatches(store,files)
    token=secrets.token_urlsafe(32);origin='http://127.0.0.1:18999';headers={'Authorization':'Bearer '+token,'Origin':origin}
    app=create_app('local',job_store=store,local_token=token,local_origin=origin)
    raw=io.BytesIO();sf.write(raw,.2*np.sin(np.arange(16000)*2*np.pi*150/16000),16000,format='WAV',subtype='FLOAT')
    ref=files.import_input(raw.getvalue(),'synthetic.wav','audio');checks=[];observed=[]
    with TestClient(app) as client:
        caps=client.get('/api/v1/capabilities').json();assert all(n in caps['algorithms'] for n in ('M01','M03','M04','M05','M06','M07','M08','M09','M14')),caps
        for cancel in (False,True,False):
            body=dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,audio=ref,config=dict(action='transform'))
            response=client.post('/api/v1/jobs/m08/create',json=body,headers=headers);assert response.status_code==201,response.text
            job=response.json();claim=store.claim('macos-qa');assert claim['id']==job['id'];e={};stop=threading.Event()
            execute_acoustic_claim(store,claim,'macos-qa',stop,on_started=(lambda pid:stop.set()) if cancel else None,process_evidence=e)
            result=store.get('local',job['id']);assert result['state']==('cancelled' if cancel else 'succeeded'),result
            if not cancel:assert result['result_manifest']['files']
            assert e.get('cleaned'),e;observed.append(e)
        checks.append('api_m08_publication_cancel_recovery')
        table=files.import_input((Path(fixtures)/'m14/public.xlsx').read_bytes(),'public.xlsx','table')
        response=client.post('/api/v1/jobs/m14/create',json=dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,table=table,config=dict(action='preview')),headers=headers)
        assert response.status_code==201,response.text;job=response.json();claim=store.claim('macos-qa');e={}
        execute_m14(store,claim,'macos-qa',threading.Event(),evidence=e);result=store.get('local',job['id']);assert result['state']=='succeeded',result;checks.append('api_m14_publication')
        denied=client.post('/api/v1/jobs/m08/create',json=body);assert denied.status_code in (401,403);checks.append('api_local_authorization')
    report=dict(success=True,checks=checks,processes=observed,capabilities=caps)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(success=True,checks=checks)));return 0
