"""Real bounded analysis, both file formats, independent EGG time axes and limits."""
import json
import sqlite3
import threading
from pathlib import Path
from uuid import uuid4
import numpy as np
import pytest
import soundfile as sf
from ptb_api.acoustic_models import AcousticConfigSnapshot
from ptb_api.acoustic_extended import AcousticExtended,JointEggSettings
from ptb_worker.acoustic_stream_child import run
from ptb_worker.io.parameter_bundle import window,BundleWriter,export_xlsx
from ptb_worker.parameter_bundle_child import run as read_bundle


def request(storage='cycles',derived=False):
    config=AcousticConfigSnapshot.model_validate(dict(settings={'min_f0':60.,'max_f0':500.},
        selection={'mode':'catalog','keys':['pF0','Intensity']},backend_policy={'reaper':'disabled','wm_f0':'irapt_then_praat'},
        extended=AcousticExtended(audio_channel=1,egg=JointEggSettings(storage=storage,derived=derived)).model_dump()))
    return dict(config=config.model_dump(),index=0,audio_sha256='a'*64,reaper_binary=None)


def signal_file(path,duration=2,rate=16000,reverse=False):
    t=np.arange(int(duration*rate))/rate
    audio=.2*np.sin(2*np.pi*200*t)+.07*np.sin(2*np.pi*400*t)
    egg=.6*np.sin(2*np.pi*100*t)
    values=np.column_stack([audio,egg] if reverse else [egg,audio])
    sf.write(path,values,rate,subtype='FLOAT')


@pytest.mark.parametrize('storage',['cycles','aligned'])
def test_real_child_and_dual_format_reopening(tmp_path,storage):
    signal_file(tmp_path/'input.wav')
    response=run(tmp_path,request(storage))
    assert response['success']
    meta=json.loads((tmp_path/'result.ptb.json').read_text('utf-8'))
    assert meta['computation_revision']=='acoustic-bounded/1' and meta['tables']['egg_cycles']['rows']>150
    view=dict(start=0.,end=2.,width=4000,parameters=['F0 - Praat','CQ','SQ','F0 - GCI'])
    first=window(tmp_path/'result.ptb.sqlite','a'*64,view)
    reopened=read_bundle(dict(source=str(tmp_path/'result.xlsx'),cache=str(tmp_path/'cache.sqlite'),name='result.xlsx',sha256='a'*64,view=view))
    assert first['columns']==reopened['columns']
    for name in view['parameters']:
        np.testing.assert_allclose(np.array(first['tracks'][name],float),np.array(reopened['tracks'][name],float),equal_nan=True,atol=1e-12)
    assert np.nanmedian(np.array(first['tracks']['F0 - Praat'],float)[:,1])==pytest.approx(200.,rel=.01)
    assert np.nanmedian(np.array(first['tracks']['F0 - GCI'],float)[:,1])==pytest.approx(100.,rel=.01)
    if storage=='cycles':assert first['tracks']['CQ'][0][0]!=first['tracks']['F0 - GCI'][0][0]


def test_window_preserves_extrema_mean_and_bounded_gaps(tmp_path):
    w=BundleWriter(tmp_path/'result.sqlite');t=np.arange(100000)/1000
    v=np.ones(len(t));v[12344]=99;v[1::2]=np.nan
    w.append('params',dict(Time_s=t,pF0=v));w.finish({'duration_s':100.,'egg':None})
    data=window(tmp_path/'result.sqlite','a'*64,dict(start=0.,end=100.,width=32,parameters=['F0 - Praat']))
    assert len(data['tracks']['F0 - Praat'])<=32*5+5
    assert data['stats']['F0 - Praat']['count']==50000
    assert data['stats']['F0 - Praat']['max']==99
    assert any(p[1]==99 for p in data['tracks']['F0 - Praat'])


def test_xlsx_literal_text_and_empty_cycle_table(tmp_path):
    w=BundleWriter(tmp_path/'first.sqlite');w.append('params',{'Time_s':np.array([0.,.1]),'text_word':np.array(['=1+2','井井'])})
    w.append('egg_cycles',{k:np.array([]) for k in ['Time_s','CQ','SQ','gF0','F0_Time_s']});w.finish({'duration_s':1.,'egg':{'storage':'cycles'}})
    export_xlsx(tmp_path/'first.sqlite',tmp_path/'a.xlsx')
    result=read_bundle(dict(source=str(tmp_path/'a.xlsx'),cache=str(tmp_path/'cache.sqlite'),name='a.xlsx',sha256='a'*64,view={'parameters':['text_word','CQ']}))
    assert result['tracks']['text_word'][0][1]=='=1+2' and not result['tracks']['CQ']


def test_legacy_config_serializes_without_new_field():
    assert 'extended' not in AcousticConfigSnapshot().model_dump()


def test_progress_atomic_publish_retries_windows_reader_sharing(tmp_path,monkeypatch):
    import ptb_worker.acoustic_stream_child as child
    original=child.os.replace;attempts=[]
    def replace(source,target):
        attempts.append(1)
        if len(attempts)<4:raise PermissionError('Windows reader sharing')
        original(source,target)
    monkeypatch.setattr(child.os,'replace',replace)
    monkeypatch.setattr(child.time,'sleep',lambda _:None)
    child.atomic(tmp_path/'status.json',{'progress':.5})
    assert len(attempts)==4 and json.loads((tmp_path/'status.json').read_text())=={'progress':.5}
    monkeypatch.setattr(child.os,'replace',lambda *_: (_ for _ in ()).throw(PermissionError('persistent denial')))
    with pytest.raises(PermissionError,match='persistent denial'):child.atomic(tmp_path/'status.json',{'progress':.6})
    assert json.loads((tmp_path/'status.json').read_text())=={'progress':.5}


def test_gci_f0_window_uses_midpoint_even_for_a_long_cycle(tmp_path):
    writer=BundleWriter(tmp_path/'result.sqlite')
    writer.append('params',{'Time_s':np.array([0.]),'pF0':np.array([100.])})
    writer.append('egg_cycles',{'Time_s':np.array([0.]),'CQ':np.array([.5]),'SQ':np.array([.1]),'gF0':np.array([.25]),'F0_Time_s':np.array([2.])})
    writer.finish({'duration_s':5.,'egg':{'storage':'cycles'}})
    value=window(tmp_path/'result.sqlite','a'*64,dict(start=1.9,end=2.1,parameters=['CQ','F0 - GCI']))
    assert value['tracks']['CQ']==[] and value['tracks']['F0 - GCI']==[[2.,.25]]


def test_derived_labels_and_channel_swap(tmp_path):
    from ptb_worker.io.parameter_bundle import label
    assert label('H1_gF0')=='H1 (gF0)' and label('H2K_gF0')=='2K (gF0)'
    signal_file(tmp_path/'input.wav',duration=1,reverse=True)
    body=request(derived=True);body['config']['extended']['channel_overrides']={'0':0}
    run(tmp_path,body)
    result=window(tmp_path/'result.ptb.sqlite','a'*64,{'parameters':['F0 - Praat','F0 - GCI']})
    assert 'CPP (gF0)' in result['columns'] and 'H1 (gF0)' in result['columns']
    assert not any(name.endswith('_gF0') for name in result['columns'])
    assert np.nanmedian(np.array(result['tracks']['F0 - Praat'],float)[:,1])==pytest.approx(200,rel=.01)
    assert np.nanmedian(np.array(result['tracks']['F0 - GCI'],float)[:,1])==pytest.approx(100,rel=.01)


def test_native_bundle_parent_slicing(tmp_path):
    from ptb_worker.acoustic_stream_segments import run_segments
    import shutil
    signal_file(tmp_path/'input.wav',duration=1);run(tmp_path,request())
    shutil.copyfile(tmp_path/'result.ptb.sqlite',tmp_path/'parent_table.bin')
    grid='''File type = "ooTextFile"
Object class = "TextGrid"
0 1 <exists> 1
"IntervalTier" "word" 0 1 1
0.2 0.7 "part"
'''
    (tmp_path/'textgrid.bin').write_text(grid,encoding='utf-8')
    result=run_segments(tmp_path,dict(audio_sha256='a'*64,audio_name='input.wav',layer='word'))
    name=next(n for n in result['files'] if n.endswith('.ptb.sqlite'))
    conn=sqlite3.connect(tmp_path/name)
    assert conn.execute('SELECT COUNT(*) FROM params').fetchone()[0]==100
    assert conn.execute('SELECT COUNT(*) FROM egg_cycles').fetchone()[0]>40
    rows=conn.execute('SELECT Time_s,Source_Time_s FROM params').fetchall()
    assert all(abs(a+.2-b)<1e-12 for a,b in rows)
    conn.close()
    original,_=sf.read(tmp_path/'input.wav');cut,_=sf.read(tmp_path/next(n for n in result['files'] if n.endswith('.wav')))
    np.testing.assert_array_equal(cut,original[3200:11200])


def test_owned_real_task(tmp_path):
    from ptb_worker.store import SQLiteJobStore,LOCAL_PROJECT
    from ptb_worker.local_acoustic_files import LocalAcousticFiles,initialize_local_files
    from ptb_worker.acoustic_batches import AcousticBatches
    from ptb_worker.acoustic_executor import execute_acoustic_claim as execute_claim
    root=Path(__file__).resolve().parents[2]
    db=tmp_path/'tasks.sqlite3'
    with sqlite3.connect((root/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as dest:source.backup(dest)
    store=SQLiteJobStore(db);cache=tmp_path/'cache';cache.mkdir();initialize_local_files(cache)
    files=LocalAcousticFiles(store,cache);batches=AcousticBatches(store,files)
    signal_file(tmp_path/'input.wav');raw=(tmp_path/'input.wav').read_bytes()
    key=files.begin_stream_input('test.wav','audio',len(raw));files.append_stream_input(key,raw);ref=files.finish_stream_input(key)
    batch=batches.submit('local',dict(project_id=LOCAL_PROJECT,operation='acoustic_analysis',idempotency_key=uuid4().hex,inputs=[{'audio':ref}],config=request()['config']))
    claim=store.claim('m01-r4');assert claim['id']==batch['summary']['items'][0]['job_id']
    execute_claim(store,claim,'m01-r4',threading.Event())
    result=store.get('local',claim['id']);assert result['state']=='succeeded',result
    assert result['result_manifest']['format_revision']=='m01-bundle/2'
    batch=batches.get('local',batch['id']);assert batch['summary']['items'][0]['progress']==1
    # Real HTTP identity/origin and the JSON UUID string contract are checked.
    from fastapi.testclient import TestClient
    from ptb_api.main import create_app
    origin='http://127.0.0.1:5177';headers={'Authorization':'Bearer '+'x'*40,'Origin':origin}
    with TestClient(create_app('local',job_store=store,local_token='x'*40,local_origin=origin),base_url=origin) as client:
        denied=client.post('/api/v1/jobs/local-inputs/stream?role=audio&name=x.wav&size=3',content=b'xxx')
        assert denied.status_code==403
        output=next(f for f in result['result_manifest']['files'] if f['name']=='result.ptb.sqlite')
        binary=files.read_result('local',output['id'],0,output['size_bytes'])
        response=client.post('/api/v1/jobs/local-inputs/stream',params={'role':'parameter_bundle','name':'result.ptb.sqlite','size':len(binary)},content=binary,headers=headers)
        assert response.status_code==200,response.text
        view=client.post('/api/v1/jobs/local-parameter-window',json={'asset_id':response.json()['asset_id'],'view':{'start':0.,'end':2.,'parameters':['CQ','F0 - GCI']}},headers=headers)
        assert view.status_code==200,view.text
        assert view.json()['tracks']['CQ'] and view.json()['streamed']
        output=next(f for f in result['result_manifest']['files'] if f['name']=='result.xlsx')
        binary=files.read_result('local',output['id'],0,output['size_bytes'])
        imported=client.post('/api/v1/jobs/local-inputs/stream',params={'role':'parameter_bundle','name':'result.xlsx','size':len(binary)},content=binary,headers=headers)
        assert imported.status_code==200,imported.text
        view=client.post('/api/v1/jobs/local-parameter-window',json={'asset_id':imported.json()['asset_id'],'view':{'start':0.,'end':2.,'parameters':['CQ','F0 - GCI']}},headers=headers)
        assert view.status_code==200,view.text
        assert view.json()['tracks']['CQ'] and view.json()['streamed']
    # Same-source lookup adds the committed SQLite sibling to the real task.
    parent=batches.parent_result('local',LOCAL_PROJECT,ref['sha256'])
    assert parent
    grid=b'File type = "ooTextFile"\nObject class = "TextGrid"\n0 2 <exists> 1\n"IntervalTier" "word" 0 2 1\n0.2 0.7 "part"\n'
    grid_ref=files.import_input(grid,'test.TextGrid','textgrid')
    sliced=batches.submit('local',dict(project_id=LOCAL_PROJECT,operation='textgrid_segment',layer='word',
        idempotency_key=uuid4().hex,inputs=[{'audio':ref,'textgrid':grid_ref,'parent_result':parent}]))
    claim=store.claim('m01-r4-slice');assert claim['id']==sliced['summary']['items'][0]['job_id']
    assert any(a['role']=='parent_table' for a in json.loads(claim['snapshot'])['input_assets'])
    execute_claim(store,claim,'m01-r4-slice',threading.Event())
    sliced_result=store.get('local',claim['id']);assert sliced_result['state']=='succeeded',sliced_result
    assert sliced_result['result_manifest']['format_revision']=='m01-bundle/2'
    assert sum(f['name'].endswith('.ptb.sqlite') for f in sliced_result['result_manifest']['files'])==1
    # Cancel the owned child and confirm the attempt cannot publish results.
    batch=batches.submit('local',dict(project_id=LOCAL_PROJECT,operation='acoustic_analysis',idempotency_key=uuid4().hex,inputs=[{'audio':ref}],config=request()['config']))
    claim=store.claim('m01-r4-cancel');stop=threading.Event();evidence={}
    execute_claim(store,claim,'m01-r4-cancel',stop,on_started=lambda pid:stop.set(),process_evidence=evidence)
    failed=store.get('local',claim['id']);assert failed['state']=='cancelled' and failed['result_manifest'] is None
    assert evidence['group_cleaned'] and evidence['temporary_payloads_cleaned']
