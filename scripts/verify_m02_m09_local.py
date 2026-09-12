"""M02/M09 real local service, bounded scientific child, durable SQLite and export.

Uses the already reviewed local test DB. No DDL or original-file modification.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4
import cv2
import numpy as np
from scipy.io import wavfile
from openpyxl import Workbook
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.local_service import LocalService
from ptb_desktop.task_bridge import TaskBridge
from ptb_worker.local_acoustic_files import initialize_local_files

ROOT=Path(__file__).resolve().parents[1]
DB=ROOT/'output/validation/p06/local-state.sqlite3'


def fixtures(folder):
    image=np.full((65,100),255,np.uint8);image[23:26]=40;image[43:46]=0
    cv2.imwrite(str(folder/'spectrogram.png'),image)
    wavfile.write(folder/'tone.wav',8000,(np.sin(2*np.pi*200*np.arange(8000)/8000)*12000).astype(np.int16))
    wb=Workbook();sheet=wb.active;sheet.append(['Time_s','pF0','rF0','Intensity','H1*-H2*','TextGrid'])
    for i in range(100):sheet.append([i/100,200+i/10,201 if i<40 or i>50 else None,60+i/100,4,'ɑ̃˥' if i<50 else 'i˩'])
    wb.save(folder/'tone.xlsx')


def main():
    out=ROOT/'output/validation/m02-m09'/('local-'+uuid4().hex);out.mkdir(parents=True)
    inputs=out/'inputs';inputs.mkdir();fixtures(inputs)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    saved=out/'saved';saved.mkdir();(saved/'reconstructed.wav').write_bytes(b'keep original')
    provider=FileProvider();directory=provider.choose('input',lambda:str(inputs));target=provider.choose('output',lambda:str(saved))
    entries={f['name']:f for f in provider.list(directory['id'])};original={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    report={'checks':[],'schema_applied':[],'success':False};ids=[]
    with sqlite3.connect(DB) as conn:
        assert not conn.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone(),'Existing active jobs'
        old=conn.execute('SELECT * FROM jobs').fetchall()
    def wait(service,job):
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            current=service.get('/api/v1/jobs/'+job['id'])
            if current['state'] not in ('queued','running','cancel_requested'):return current
            time.sleep(.15)
        raise AssertionError('Job timeout')
    try:
        with LocalService(DB,local_files_root=cache,reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe') as service:
            bridge=TaskBridge(provider,service)
            table=bridge.invoke(dict(op='parameters',id=entries['tone.xlsx']['id']))
            assert table['rows'][45][2] is None and table['rows'][0][-1]=='ɑ̃˥';report['checks'].append('actual native file grant and bounded table API')
            request=dict(op='reconstruct',id=entries['spectrogram.png']['id'],config=dict(time_end=.2,n_iter=3,seed=12),key=uuid4().hex)
            job=bridge.invoke(request);ids.append(job['id']);assert bridge.invoke(request)['id']==job['id']
            job=wait(service,job);assert job['state']=='succeeded',job
            report['checks'].append('real immutable image admission, idempotency, Windows child and complete publication')
            result=bridge.invoke(dict(op='save_job',id=job['id'],directory=target['id']));assert result['count']==4
            assert (saved/'reconstructed.wav').read_bytes()==b'keep original'
            before={p.name:p.read_bytes() for p in saved.iterdir()}
            assert bridge.invoke(dict(op='save_job',id=job['id'],directory=target['id']))['saved']==result['saved']
            assert before=={p.name:p.read_bytes() for p in saved.iterdir()}
            audio=next(saved/n for n in result['saved'] if n.endswith('.wav'));rate,samples=wavfile.read(audio)
            metadata=json.loads((saved/'reconstruction.ptb.json').read_text('utf-8'))
            assert rate==44100 and len(samples)==metadata['samples'] and metadata['seed']==12
            for f in job['result_manifest']['files']:assert bridge.invoke(dict(op='result',job=job['id'],id=f['id']))['sha256']==f['sha256']
            report['checks'].append('WAV/PNG/JSON exact readback, hashed previews, idempotent nonoverwriting native export')
            # Stop the owned worker before admission to exercise queued cancellation and restart.
        with LocalService(DB,local_files_root=cache,reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe') as service:
            bridge=TaskBridge(provider,service);assert bridge.invoke(dict(op='job',id=ids[0]))['state']=='succeeded'
            failed=bridge.invoke(dict(op='reconstruct',id=entries['spectrogram.png']['id'],config=dict(time_end=.2,n_iter=1,corners=[dict(x=0,y=0)]*4),key=uuid4().hex));ids.append(failed['id'])
            failed=wait(service,failed);assert failed['state']=='failed' and failed['error_code']=='invalid_image_corners',failed
            retry=bridge.invoke(dict(op='retry',id=failed['id'],key=uuid4().hex));ids.append(retry['id']);assert retry['retry_of']==failed['id'];assert wait(service,retry)['state']=='failed'
            cancel=bridge.invoke(dict(op='reconstruct',id=entries['spectrogram.png']['id'],config=dict(time_end=1,n_iter=128),key=uuid4().hex));ids.append(cancel['id'])
            bridge.invoke(dict(op='cancel_job',id=cancel['id']));assert wait(service,cancel)['state']=='cancelled'
            report['checks'].append('service restart preserves success; invalid quad, immutable retry and cancellation remain explicit')
        state=json.loads((cache/'.ptb-local.json').read_text('utf-8'))
        assert not [a for a in state['assets'].values() if a['state']!='deleted' and a['kind']=='temporary']
        with sqlite3.connect(DB) as conn:
            for row in old:assert conn.execute('SELECT * FROM jobs WHERE id=?',(row[0],)).fetchone()==row
        assert original=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        report.update(success=True,jobs=ids,inputs_preserved=True,existing_rows_preserved=True,temporary_files_remaining=0)
    finally:
        provider.close();(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out/'report.json')


if __name__=='__main__':main()
