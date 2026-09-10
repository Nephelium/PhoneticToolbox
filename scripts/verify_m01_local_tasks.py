"""Approved fixed SQLite + new UUID synthetic files; actual local API and worker."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import time
import sys
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.local_service import LocalService
from ptb_desktop.task_bridge import TaskBridge
from ptb_worker.local_acoustic_files import initialize_local_files

ROOT=Path(__file__).resolve().parents[1]
DB=ROOT/'output/validation/p06/local-state.sqlite3'


def verify():
    out=ROOT/'output/validation/m01'/('local-tasks-'+uuid4().hex);out.mkdir()
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    input_dir=out/'inputs';input_dir.mkdir();target=out/'saved';target.mkdir()
    fs=16000;t=np.arange(fs)/fs
    signal=np.column_stack(((12000*np.sin(2*np.pi*200*t)).astype(np.int16),(6000*np.sin(2*np.pi*200*t)).astype(np.int16)))
    wavfile.write(input_dir/'tone.wav',fs,signal)
    (input_dir/'tone.TextGrid').write_text('File type = "ooTextFile short"\n"TextGrid"\n0\n1\n<exists>\n1\n"IntervalTier"\n"syllable"\n0\n1\n3\n0\n.2\n"sil"\n.2\n.6\n"a"\n.6\n1\n"eps"\n',encoding='utf-8')
    (target/'tone.xlsx').write_bytes(b'existing research file')
    with sqlite3.connect(DB) as conn:
        assert not conn.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone(),'Existing active tasks'
        old=conn.execute('SELECT * FROM jobs').fetchall()
    provider=FileProvider();directory=provider.choose('input',lambda:str(input_dir));destination=provider.choose('output',lambda:str(target))
    entries={f['kind']:f['id'] for f in provider.list(directory['id'])};checks={};ids=[]
    options=dict(local_files_root=cache,reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe')
    def record():
        (out/'report.json').write_text(json.dumps(dict(checks=checks,batches=ids,cache=str(cache)),indent=2)+'\n',encoding='utf-8')
    def wait(bridge,batch):
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            batch=bridge.invoke(dict(op='get',id=batch['id']))
            if batch['summary']['closed']:return batch
            time.sleep(.2)
        raise AssertionError('Task did not finish')
    try:
        with LocalService(DB,**options) as service:
            bridge=TaskBridge(provider,service)
            batch=bridge.invoke(dict(op='submit',operation='acoustic_analysis',inputs=[dict(audio=entries['audio'],textgrid=entries['textgrid'])],
                config=dict(selection=dict(mode='catalog',keys=['pF0','rF0','Intensity']),backend_policy=dict(reaper='native_required')),layer=None,idempotency_key=uuid4().hex))
            ids.append(batch['id']);record();batch=wait(bridge,batch)
            job=bridge.invoke(dict(op='job',id=batch['summary']['items'][0]['job_id']))
            (out/'analysis-job.json').write_text(json.dumps(job,indent=2),encoding='utf-8')
            assert batch['summary']['complete'],job['error_code']
            saved=bridge.invoke(dict(op='save',id=batch['id'],directory=destination['id']));assert saved['count']==3
            assert (target/'tone.xlsx').read_bytes()==b'existing research file'
            before={p.name:p.read_bytes() for p in target.iterdir()}
            assert bridge.invoke(dict(op='save',id=batch['id'],directory=destination['id']))['saved']==saved['saved']
            assert before=={p.name:p.read_bytes() for p in target.iterdir()}
            checks['real_science_and_nonoverwriting_idempotent_save']=True;record()
            wire=json.loads((target/'tone.ptb.json').read_text('utf-8'))
            (out/'analysis-wire.json').write_text(json.dumps(wire,indent=2),encoding='utf-8')
            assert any(b['actual']=='native_reaper' for b in wire['metadata']['backends'])
            for column in wire['numeric']:
                if column['key']=='pF0':
                    finite=[x for x in column['values'] if x is not None];assert finite and abs(float(np.median(finite))-200)<2,column['key']
            parent=bridge.invoke(dict(op='parent',id=entries['audio']));assert parent
            cut=bridge.invoke(dict(op='submit',operation='textgrid_segment',inputs=[dict(audio=entries['audio'],textgrid=entries['textgrid'],parent_result=parent)],
                config=None,layer='syllable',idempotency_key=uuid4().hex))
            ids.append(cut['id']);record();cut=wait(bridge,cut)
            assert cut['summary']['complete'],bridge.invoke(dict(op='job',id=cut['summary']['items'][0]['job_id']))
            saved_cut=bridge.invoke(dict(op='save',id=cut['id'],directory=destination['id']));assert saved_cut['count']==4
            name=next(name for name in saved_cut['saved'] if name.endswith('.wav'))
            rate,data=wavfile.read(target/name);assert rate==fs;np.testing.assert_array_equal(data,signal[3200:9600])
            checks['actual_parent_parameter_cut_and_stereo_samples']=True;record()
            # A failed later download rolls back only files newly created by this attempt.
            rollback=out/'rollback';rollback.mkdir();(rollback/'keep.txt').write_text('keep',encoding='utf-8')
            rollback_grant=provider.choose('output',lambda:str(rollback));real=service.binary;calls=[0]
            def failure(*args,**kwargs):
                calls[0]+=1
                if calls[0]==2:raise OSError('Injected second download failure')
                return real(*args,**kwargs)
            service.binary=failure
            try:bridge.invoke(dict(op='save',id=batch['id'],directory=rollback_grant['id']))
            except OSError:pass
            else:raise AssertionError('Expected interrupted download')
            finally:service.binary=real
            assert sorted(p.name for p in rollback.iterdir())==['keep.txt']
            checks['partial_export_rollback_preserves_existing']=True;record()
            # Verify REAPER and all 80 outputs against frozen original-v2 data,
            # not an assumed pure-sine REAPER pitch estimate.
            from baseline_support import RECIPES,create_fixture,load_json,compare
            sys.path.insert(0,str(ROOT/'tests/support'))
            from m01_science import pack
            from ptb_api.acoustic_models import AcousticSettings
            source=create_fixture(input_dir,RECIPES[0]);old_golden=load_json(ROOT/'tests/fixtures/m01/GUI-ALL.json.gz')['scientific']
            source_id=next(f['id'] for f in provider.list(directory['id']) if f['name']==source.name)
            golden=bridge.invoke(dict(op='submit',operation='acoustic_analysis',inputs=[dict(audio=source_id)],layer=None,idempotency_key=uuid4().hex,
                config=dict(settings={k:old_golden['config'][k] for k in AcousticSettings.model_fields},selection=dict(mode='catalog',keys=old_golden['config']['selected_parameter_keys']),backend_policy=dict(reaper='native_required'))))
            ids.append(golden['id']);record();golden=wait(bridge,golden);assert golden['summary']['complete'],golden
            bridge.invoke(dict(op='save',id=golden['id'],directory=destination['id']))
            restored=json.loads((target/(source.stem+'.ptb.json')).read_text('utf-8'))
            tracks={'Time_s':pack(np.array(restored['times_s']))}
            for c in restored['numeric']:
                tracks[c['key']]=pack(np.array([v if mask==0 else (np.nan,np.inf,-np.inf)[mask-1] for v,mask in zip(c['values'],c['nonfinite'])]))
            for c in restored['text']:tracks[c['key']]=pack(np.array(c['values'],dtype=object))
            assert restored['column_order']==old_golden['columns'] and restored['times_s']==old_golden['time_axis']
            differences=compare(old_golden['tracks'],tracks);assert not differences,differences
            checks['all_80_against_original_v2_golden']=True;record()
        assert service.exit_code==0
        with LocalService(DB,**options) as reopened:
            bridge=TaskBridge(provider,reopened)
            assert all(bridge.invoke(dict(op='get',id=key))['summary']['complete'] for key in ids)
            assert bridge.invoke(dict(op='save',id=ids[0],directory=destination['id']))['count']==3
        assert reopened.exit_code==0
        checks['owned_service_restart_and_results']=True
        with sqlite3.connect(DB) as conn:assert set(old)<=set(conn.execute('SELECT * FROM jobs').fetchall())
        ledger=json.loads((cache/'.ptb-local.json').read_text('utf-8'))
        assert not any(a['reserved_bytes'] or a['state'] in ('uploading','deleting') for a in ledger['assets'].values())
        checks['old_rows_and_no_scratch_reservations']=True;record()
        print(json.dumps(dict(output=str(out),checks=checks)))
    finally:
        record();provider.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--approved-m01-synthetic-tests',action='store_true');a=p.parse_args()
    if not a.approved_m01_synthetic_tests:p.error('Explicit synthetic test approval required')
    verify()
