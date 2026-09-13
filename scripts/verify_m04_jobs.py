"""M04-C actual local HTTP, owned child, recovery and non-overwriting save. No DDL."""
import base64
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
import struct
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.local_service import LocalService
from ptb_desktop.task_bridge import TaskBridge
from ptb_worker.local_acoustic_files import initialize_local_files, LocalAcousticFiles
from ptb_worker.store import SQLiteJobStore, LOCAL_PROJECT, JobError
from ptb_worker.lpc_jobs import submit
from ptb_worker.acoustic_executor import execute_acoustic_claim
from ptb_api.quota import StorageError

ROOT=Path(__file__).resolve().parents[1]


def dead(pid):
    kernel=ctypes.WinDLL('kernel32');kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return True
    code=ctypes.c_ulong();kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    try:
        assert kernel.GetExitCodeProcess(handle,ctypes.byref(code))
        return code.value!=259
    finally:kernel.CloseHandle(handle)


def main():
    out=ROOT/'output/validation/m04-jobs'/uuid4().hex;out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
        source.backup(target);old=source.execute('SELECT * FROM jobs').fetchall()
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    t=np.arange(96000)/48000;audio=.3*np.sin(2*np.pi*150*t)+.1*np.sin(2*np.pi*800*t)
    wavfile.write(inputs/'ɑ̃˥ LPC.wav',48000,np.column_stack([audio,audio*.5]))
    wavfile.write(inputs/'silent.wav',48000,np.zeros(96000))
    # Frozen short-format TextGrid syntax; two intervals exercise old end-label rule.
    (inputs/'labels.TextGrid').write_text('File type = "ooTextFile"\nObject class = "TextGrid"\n\n0\n2\n<exists>\n1\n"IntervalTier"\n"phones"\n0\n2\n2\n0\n1\n"ɑ̃˥"\n1\n2\n"末"\n',encoding='utf-8')
    original={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
    provider=FileProvider();directory=provider.choose('input',lambda:str(inputs));destination=provider.choose('output',lambda:str(saved))
    entries={f['name']:f for f in provider.list(directory['id'])}
    report=dict(success=False,schema_applied=[],checks=[],jobs=[])
    def wait(service,job):
        end=time.monotonic()+50
        while time.monotonic()<end:
            current=service.get('/api/v1/jobs/'+job['id'])
            if current['state'] not in ('queued','running','cancel_requested'):return current
            time.sleep(.1)
        raise AssertionError('Task did not settle')
    try:
        with LocalService(db,local_files_root=cache) as service:
            bridge=TaskBridge(provider,service)
            assert bridge.invoke(dict(op='lpc_fonts',font={}))['available']
            configs=[dict(roi_start=.1,roi_end=.2),dict(roi_start=1.5,roi_end=1.6,dynamic_y=True),dict(roi_end=1.,order=200)]
            for config in configs:
                request=dict(op='lpc',id=entries['ɑ̃˥ LPC.wav']['id'],config=config,key=uuid4().hex)
                started=time.monotonic();job=bridge.invoke(request)
                assert bridge.invoke(request)['id']==job['id']
                done=wait(service,job);assert done['state']=='succeeded',done
                report['jobs'].append(dict(id=job['id'],seconds=time.monotonic()-started))
                returned={}
                for file in done['result_manifest']['files']:
                    raw=base64.b64decode(bridge.invoke(dict(op='result',job=job['id'],id=file['id']))['base64'])
                    assert hashlib.sha256(raw).hexdigest()==file['sha256'];returned[file['name']]=raw
                meta=json.loads(returned['lpc.ptb.json']);first,last=meta['selection']['start_sample'],meta['selection']['end_sample']
                rate,values=wavfile.read(io.BytesIO(returned['lpc_AUDIO.wav']))
                expected=np.mean(np.column_stack([audio,audio*.5])[first:last],axis=1)
                assert rate==48000 and values.tobytes()==expected.tobytes()
                png=returned['lpc_SPECTRUM.png'];assert png[:8]==b'\x89PNG\r\n\x1a\n'
                assert struct.unpack('>II',png[16:24])==(2400,1350)
                assert b'pHYs'+struct.pack('>IIB',11811,11811,1) in png
                name=meta['export_names']['lpc_SPECTRUM.png'];(saved/name).write_bytes(b'preserve old result')
                exported=bridge.invoke(dict(op='save_job',id=job['id'],directory=destination['id']))
                assert exported['count']==3 and (saved/name).read_bytes()==b'preserve old result'
                assert bridge.invoke(dict(op='save_job',id=job['id'],directory=destination['id']))['saved']==exported['saved']
            report['checks'].append('actual HTTP/MKL/PNG/WAV/JSON; max 48000 samples; tail ROI; same-name save and repeat')
        with LocalService(db,local_files_root=cache) as service:
            assert all(service.get('/api/v1/jobs/'+j['id'])['state']=='succeeded' for j in report['jobs'])
        store=SQLiteJobStore(db);files=LocalAcousticFiles(store,cache)
        ref=files.import_input((inputs/'ɑ̃˥ LPC.wav').read_bytes(),'ɑ̃˥ LPC.wav','audio')
        grid=files.import_input((inputs/'labels.TextGrid').read_bytes(),'labels.TextGrid','textgrid')
        def admit(config=None,reference=None,textgrid=None):
            return submit(store,'local',dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,audio=reference or ref,textgrid=textgrid,config=config or {'roi_end':.1}))
        def execute(job,on_started=None):
            claim=store.claim('m04-check');assert claim['id']==job['id']
            execute_acoustic_claim(store,claim,'m04-check',threading.Event(),on_started=on_started)
            return store.get('local',job['id'])
        labelled=admit(dict(roi_end=.1,tier_name='phones'),textgrid=grid);assert execute(labelled)['state']=='succeeded'
        manifest=store.get('local',labelled['id'])['result_manifest'];meta=next(f for f in manifest['files'] if f['name']=='lpc.ptb.json')
        assert json.loads(files.read_result('local',meta['id'],0,meta['size_bytes']))['label']=='ɑ̃˥'
        report['checks'].append('TextGrid IPA label and persisted font snapshot; restart restores successful tasks')
        for config,reference,code in [({'roi_end':.1},files.import_input((inputs/'silent.wav').read_bytes(),'silent.wav','audio'),'lpc_solver_failed'),
                ({'roi_end':1.1},None,'lpc_roi_budget'),({'roi_end':.0005},None,'lpc_segment_too_short'),
                ({'roi_end':.1,'font':{'latin':'PTB Missing Font'}},None,'font_unavailable')]:
            done=execute(admit(config,reference));assert done['state']=='failed' and done['error_code']==code,done
        cancelled=admit();store.cancel('local',cancelled['id']);retry=store.retry('local',cancelled['id'],uuid4().hex)
        assert retry['retry_of']==cancelled['id'];pids=[]
        def cancel(pid):pids.append(pid);store.cancel('local',retry['id'])
        assert execute(retry,cancel)['state']=='cancelled' and all(dead(p) for p in pids)
        replacement=store.retry('local',retry['id'],uuid4().hex);assert execute(replacement)['state']=='succeeded'
        report['checks'].append('queued/live-child cancel; PID exited; immutable retry succeeds; semantic failures publish nothing')
        failed=admit();write=files.write;count=[0]
        def fail_write(identity,asset_id,offset,raw):
            with files.locked() as state:kind=state['assets'][asset_id]['kind']
            if kind=='result':
                count[0]+=1
                if count[0]==2:raise StorageError('storage_write_failed',503)
            return write(identity,asset_id,offset,raw)
        files.write=fail_write
        try:assert execute(failed)['state']=='failed' and count[0]==2
        finally:files.write=write
        # Same real executor deadline with a controlled stalled owned child.
        import ptb_worker.lpc_runtime as runtime
        command=runtime.command;pids=[];timeout_job=admit();start=time.monotonic()
        runtime.command=lambda request,pipe:[sys.executable,'-c','import time; time.sleep(40)',pipe]
        try:done=execute(timeout_job,lambda pid:pids.append(pid))
        finally:runtime.command=command
        report['timeout_seconds']=time.monotonic()-start
        assert done['state']=='failed' and done['error_code']=='deadline_exceeded',done
        assert 29<=report['timeout_seconds']<38 and all(dead(p) for p in pids)
        report['checks'].append('actual 30-second deadline kills owned stalled child; partial write rollback')
        # Expired worker lease cannot finish or publish.
        stale=admit();claim=store.claim('m04-stale');identity=(claim['id'],'m04-stale',claim['generation'])
        pending=files.output(identity,'lpc_SPECTRUM.png','result',1);files.write(identity,pending['id'],0,b'x')
        with store.transaction() as tx:tx.execute('UPDATE jobs SET lease_until=0 WHERE id=?',(claim['id'],))
        store.recover();files.recover();assert store.get('local',stale['id'])['state']=='interrupted'
        try:files.seal(identity,pending['id'])
        except StorageError:pass
        else:raise AssertionError('Stale worker published')
        state=json.loads((cache/'.ptb-local.json').read_text('utf-8'))
        assert all(a['state']=='deleted' for a in state['assets'].values() if a['kind']=='temporary')
        for j in store.list('local',LOCAL_PROJECT):
            if j['operation']=='lpc_analysis' and j['state']!='succeeded':
                assert not j['result_manifest']
                assert all(a['state']=='deleted' for a in state['assets'].values() if a['job_id']==j['id'])
        with sqlite3.connect(db) as conn:
            for row in old:assert conn.execute('SELECT * FROM jobs WHERE id=?',(row[0],)).fetchone()==row
        assert original=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        report.update(success=True,old_rows_preserved=True,input_hashes_preserved=True,temporary_files_remaining=0)
    finally:
        provider.close();(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out/'report.json',flush=True)


if __name__=='__main__':main()
