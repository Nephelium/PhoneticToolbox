"""M03-R7 high-cycle-rate 30min CSV, measured native resources and stream save."""
import hashlib,json,os,sqlite3,threading,time,wave
from contextlib import closing
from pathlib import Path
from verification_artifacts import verification_run
from uuid import uuid4
import numpy as np
from ptb_worker.store import SQLiteJobStore,LOCAL_PROJECT
from ptb_worker.local_acoustic_files import LocalAcousticFiles,initialize_local_files
from ptb_worker.egg_jobs import submit
from ptb_worker.acoustic_executor import execute_acoustic_claim
from ptb_desktop.local_service import LocalService
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.task_bridge import TaskBridge
ROOT=Path(__file__).resolve().parents[1]
def main():
    with verification_run('m03-r7', 'large-csv', '8 kHz stereo PCM16, 1800 s; deterministic 510 Hz with 1020/1530 Hz harmonics, scale 16000; isolated database/cache/CSV') as (out, scratch):
        path=scratch/'510Hz-30min.wav';fs=8000
        with wave.open(str(path),'wb') as f:
            f.setparams((2,2,fs,0,'NONE','not compressed'))
            for first in range(0,1800,20):
                t=(np.arange(fs*20)+fs*first)/fs
                values=np.column_stack((np.sin(2*np.pi*510*t)+.2*np.sin(2*np.pi*1020*t),np.sin(2*np.pi*510*t)+.1*np.sin(2*np.pi*1530*t)))
                f.writeframes((values*16000).astype('<i2').tobytes())
        db=scratch/'jobs.sqlite3'
        with closing(sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True)) as src,closing(sqlite3.connect(db)) as dst:src.backup(dst)
        cache=scratch/'cache';cache.mkdir();initialize_local_files(cache)
        store=SQLiteJobStore(db);files=LocalAcousticFiles(store,cache)
        key=files.begin_stream_input(path.name,'audio',path.stat().st_size)
        with path.open('rb') as f:
            for raw in iter(lambda:f.read(1048576),b''):files.append_stream_input(key,raw)
        reference=files.finish_stream_input(key)
        job=submit(store,'local',dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,audio=reference,config=dict(mode='single',f0_policy='audio-f0/2')))
        claim=store.claim('M03-R7-owned-QA');os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
        evidence={};report=dict(success=False,resources=evidence,schema_applied=[]);started=time.monotonic()
        try:
            execute_acoustic_claim(store,claim,'M03-R7-owned-QA',threading.Event(),process_evidence=evidence)
            done=store.get('local',job['id']);assert done['state']=='succeeded',done
            manifest=done['result_manifest'];assert manifest['format_revision']=='m03/2'
            csv=next(f for f in manifest['files'] if f['name']=='egg_DATA.csv');assert csv['size_bytes']>64000000
            saved=scratch/'saved';saved.mkdir();provider=FileProvider();destination=provider.choose('output',lambda:str(saved))
            try:
                with LocalService(db,local_files_root=cache) as service:
                    bridge=TaskBridge(provider,service);result=bridge.invoke(dict(op='save_job',id=job['id'],directory=destination['id']))
                    assert result['count']==5
            finally:provider.close()
            csv_path=next(saved.glob('*.csv'))
            with csv_path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==csv['sha256']
            with csv_path.open('r',encoding='utf-8') as f:
                lines=0;last=''
                for line in f:lines+=1;last=line
            assert float(last.split(',')[0])>1799
            state=json.loads((cache/'.ptb-local.json').read_text('utf-8'))
            assert not [a for a in state['assets'].values() if a['kind']=='temporary' and a['state']!='deleted']
            report.update(success=True,csv_bytes=csv['size_bytes'],csv_rows=lines-1,seconds=time.monotonic()-started,temporary_files_remaining=0)
        finally:
            (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(out,flush=True)

if __name__=='__main__':main()
