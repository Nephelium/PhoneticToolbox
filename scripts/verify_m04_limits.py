"""M04 maximum decoded file and selected tail through the actual bounded child."""
import io
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
from ptb_worker.local_acoustic_files import initialize_local_files, LocalAcousticFiles
from ptb_worker.store import SQLiteJobStore, LOCAL_PROJECT
from ptb_worker.lpc_jobs import submit
from ptb_worker.acoustic_executor import execute_acoustic_claim

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'output/validation/m04-limits'/uuid4().hex;out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(db) as dest:
        src.backup(dest)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    store=SQLiteJobStore(db);files=LocalAcousticFiles(store,cache)
    os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
    report=[];samples=np.random.default_rng(404).integers(-30000,30000,size=8_000_001,dtype=np.int16)
    for count,expected in [(8_000_000,'succeeded'),(8_000_001,'failed')]:
        stream=io.BytesIO();wavfile.write(stream,8000,samples[:count]);raw=stream.getvalue()
        ref=files.import_input(raw,'1000-second.wav','audio')
        body=dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,audio=ref,config=dict(roi_start=999.9,roi_end=1000.))
        job=submit(store,'local',body);claim=store.claim('m04-limits');assert claim['id']==job['id']
        start=time.monotonic();execute_acoustic_claim(store,claim,'m04-limits',threading.Event())
        done=store.get('local',job['id']);assert done['state']==expected,done
        if expected=='succeeded':
            meta=next(f for f in done['result_manifest']['files'] if f['name']=='lpc.ptb.json')
            data=json.loads(files.read_result('local',meta['id'],0,meta['size_bytes']))
            assert data['sample_count']==8_000_000 and data['selection']['end_sample']==8_000_000
            assert data['selection']['start_sample']==7_999_200 and len(data['spectrum']['magnitude_db'])==1024
        else:assert done['error_code']=='lpc_input_budget' and not done['result_manifest']
        report.append(dict(sample_count=count,input_bytes=len(raw),state=done['state'],error=done['error_code'],seconds=time.monotonic()-start))
    state=json.loads((cache/'.ptb-local.json').read_text('utf-8'))
    assert all(a['state']=='deleted' for a in state['assets'].values() if a['kind']=='temporary')
    (out/'report.json').write_text(json.dumps(dict(success=True,cases=report,schema_applied=[],temporary_files_remaining=0),indent=2),encoding='utf-8')
    print(out/'report.json')


if __name__=='__main__':main()
