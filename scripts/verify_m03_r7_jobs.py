"""M03-R7 public long file and ten-second IF through real local jobs and saves."""
import argparse,base64,hashlib,io,json,os,sqlite3,time
from contextlib import closing
from pathlib import Path
from verification_artifacts import verification_run
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
from PyQt6.QtGui import QImage
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.local_service import LocalService
from ptb_desktop.task_bridge import TaskBridge
from ptb_worker.local_acoustic_files import initialize_local_files

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True,type=Path);args=parser.parse_args()
    with verification_run('m03-r7', 'jobs', 'External 48 kHz 1800 s input retained; generated 96 kHz stereo PCM16 10 s, 150 Hz harmonics, scale 16000; isolated database/cache/exports') as (out, scratch):
        db=scratch/'jobs.sqlite3'
        with closing(sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True)) as src,closing(sqlite3.connect(db)) as dst:
            assert not src.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
            src.backup(dst);old=src.execute('SELECT * FROM jobs').fetchall()
        cache=scratch/'cache';cache.mkdir();initialize_local_files(cache)
        saved=scratch/'saved';saved.mkdir();inputs=scratch/'inputs';inputs.mkdir()
        t=np.arange(960000)/96000
        y=np.column_stack((np.sin(2*np.pi*150*t)+.2*np.sin(2*np.pi*300*t),np.sin(2*np.pi*150*t)+.1*np.sin(2*np.pi*450*t)))
        wavfile.write(inputs/'96k-ten-seconds.wav',96000,(y*16000).astype(np.int16))
        os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
        provider=FileProvider();report=dict(success=False,jobs=[],schema_applied=[])
        grant=provider.choose('input',lambda:str(args.source.resolve().parent));file=next(f for f in provider.list(grant['id']) if f['name']==args.source.name)
        grant96=provider.choose('input',lambda:str(inputs));file96=provider.list(grant96['id'])[0]
        destination=provider.choose('output',lambda:str(saved));source_hash=hashlib.file_digest(args.source.open('rb'),'sha256').hexdigest()
        def wait(service,job):
            deadline=time.monotonic()+240
            while time.monotonic()<deadline:
                result=service.get('/api/v1/jobs/'+job['id'])
                if result['state'] not in ('queued','running','cancel_requested'):return result
                time.sleep(.1)
            raise AssertionError('job timeout')
        try:
            with LocalService(db,local_files_root=cache,reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe') as service:
                bridge=TaskBridge(provider,service)
                for label,entry,config,rate,frames in [
                    ('full-thirty-minutes',file,dict(mode='single',roi_start=0,roi_end=None,f0_policy='audio-f0/2'),48000,None),
                    ('ten-second-tail',file,dict(mode='inverse',roi_start=1790,roi_end=1800,f0_policy='audio-f0/2'),48000,480000),
                    ('96k-ten-seconds',file96,dict(mode='inverse',roi_start=0,roi_end=10,f0_policy='audio-f0/2'),96000,960000)]:
                    request=dict(op='egg',id=entry['id'],config=config,key=uuid4().hex)
                    started=time.monotonic();job=bridge.invoke(request)
                    assert bridge.invoke(request)['id']==job['id']
                    done=wait(service,job);assert done['state']=='succeeded',done
                    item=dict(label=label,job=job['id'],seconds=time.monotonic()-started,files=[]);report['jobs'].append(item)
                    for f in done['result_manifest']['files']:
                        raw=base64.b64decode(bridge.invoke(dict(op='result',job=job['id'],id=f['id']))['base64'])
                        assert len(raw)==f['size_bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256']
                        item['files'].append(dict(name=f['name'],size=len(raw)))
                        if f['name'].endswith('.wav'):
                            fs,audio=wavfile.read(io.BytesIO(raw));assert fs==rate and len(audio)==frames and np.isfinite(audio).all()
                        elif f['name'].endswith('.png'):assert not QImage.fromData(raw,'PNG').isNull()
                        elif f['name'].endswith('.json'):
                            meta=json.loads(raw)
                            assert meta['sample_rate_hz']==rate
                            if frames:assert meta['inverse']['sample_count']==frames
                            else:assert meta['bounded']['revision']=='egg-bounded/2'
                        elif f['name'].endswith('.csv'):
                            lines=raw.decode().splitlines();assert float(lines[-1].split(',')[0])>1799
                            item['csv_rows']=len(lines)-1
                    result=bridge.invoke(dict(op='save_job',id=job['id'],directory=destination['id']))
                    assert result['count']==len(item['files'])
                    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            assert hashlib.file_digest(args.source.open('rb'),'sha256').hexdigest()==source_hash
            with closing(sqlite3.connect(db)) as conn:
                for row in old:assert conn.execute('SELECT * FROM jobs WHERE id=?',(row[0],)).fetchone()==row
            state=json.loads((cache/'.ptb-local.json').read_text('utf-8'))
            assert not [a for a in state['assets'].values() if a['state']!='deleted' and a['kind']=='temporary']
            report.update(success=True,old_rows_preserved=True,source_preserved=True,temporary_files_remaining=0)
        finally:
            provider.close();(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(out,flush=True)

if __name__=='__main__':main()
