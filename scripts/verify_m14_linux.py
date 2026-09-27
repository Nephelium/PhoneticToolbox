"""M14 authorized Linux qualification. Uses a new schema COPY; never DDL.

Production M14 admission remains closed until P11 registers and qualifies it.
The core/child probe is qualification, not a bypass for application requests.
"""
import argparse,json,os,sys,hashlib,sqlite3,secrets,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--serve',action='store_true');args=p.parse_args()
    out=ROOT/'evidence'/('linux-'+str(time.time_ns()));out.mkdir(parents=True)
    if args.serve:
        from ptb_worker.store import SQLiteJobStore,LOCAL_PROJECT
        from ptb_worker.local_acoustic_files import LocalAcousticFiles,initialize_local_files
        from ptb_worker.acoustic_batches import AcousticBatches
        from ptb_api.main import create_app
        from fastapi.testclient import TestClient
        from starlette.staticfiles import StaticFiles
        import uvicorn
        db=out/'jobs.sqlite3'
        with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(db) as dst:src.backup(dst)
        store=SQLiteJobStore(db);store.check_schema();cache=out/'cache';cache.mkdir();initialize_local_files(cache);files=LocalAcousticFiles(store,cache);AcousticBatches(store,files)
        token=secrets.token_urlsafe(32);origin='http://127.0.0.1:18941';app=create_app('local',job_store=store,local_token=token,local_origin=origin)
        with TestClient(app) as client:
            body=dict(project_id=LOCAL_PROJECT,idempotency_key='m14-linux-unavailable',table=dict(asset_id='00000000-0000-4000-8000-000000000002',sha256='0'*64),config=dict(action='preview'))
            rejected=client.post('/api/v1/jobs/m14/create',json=body,headers={'Authorization':'Bearer '+token,'Origin':origin})
            assert rejected.status_code==503 and rejected.json()['detail']=='m14_runtime_unavailable'
            assert 'M14' not in client.get('/api/v1/capabilities').json()['algorithms']
            (out/'service-report.json').write_text(json.dumps(dict(success=True,scope='native Linux formal API capability/admission rejection only; M14 durable execution not enabled',status=rejected.status_code,response=rejected.json()),indent=2))
        app.mount('/',StaticFiles(directory=ROOT/'frontend/dist',html=True),name='workbench')
        server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=18941,access_log=False,log_level='warning'))
        timer=threading.Timer(180,lambda:setattr(server,'should_exit',True));timer.daemon=True;timer.start()
        print('M14_OWNED_LOOPBACK_READY',flush=True);server.run();return
    from ptb_worker.native.posix import run_bounded
    from ptb_worker.io.limits import Limits
    limits=Limits(process_bytes=512000000,output_bytes=16000000,timeout_seconds=60)
    prefix=['/usr/bin/env','PYTHONPATH='+os.environ['PYTHONPATH'],sys.executable];e={}
    raw=run_bounded(prefix+['-m','pytest','-c','tests/pytest.ini','tests/parity/test_phonology_induction.py','backend/tests/test_m14.py','-q','--tb=short'],b'',ROOT,limits,evidence=e)
    (out/'tests.txt').write_bytes(raw);(out/'process.json').write_text(json.dumps(e,indent=2));print(raw.decode(),flush=True)
    from ptb_worker.m14_jobs import execute
    from ptb_worker.m14_executor import unpack
    source=(ROOT/'tests/fixtures/m14/public.xlsx').read_bytes();cfg=dict(action='preview',skip_first_row=True,consonant_only_as_zero_initial=True);preview=json.loads(execute(source,'public.xlsx',cfg)['m14-preview.json']);cfg.update(action='export',settings=preview['config'],font=dict(schema_version='font/1',zh='Noto Sans SC',latin='DejaVu Sans',ipa='Doulos SIL',size_px=14))
    sha=hashlib.sha256(source).hexdigest();request=out/'child.request';request.write_bytes(json.dumps(dict(config=cfg,name='public.xlsx',sha256=sha)).encode()+b'\n'+source);e={}
    raw=run_bounded(prefix+['-m','ptb_worker.m14_child',str(request),'/dev/stdout'],b'',ROOT,limits,evidence=e)
    values=unpack(raw,sha,'export')
    for name,data in values:(out/name).write_bytes(data)
    (out/'child-process.json').write_text(json.dumps(e,indent=2));print('FIXED_CHILD_THREE_FILES',len(values),out,flush=True)

if __name__=='__main__':main()
