"""Recording workflow regression: explicit scope plans and Windows sharing races."""
import copy
import os
import threading
import numpy as np
import pytest
import soundfile as sf
from ptb_desktop.recording import storage
from ptb_desktop.recording.jobs import export_items,export_audio


def fixture_project(tmp_path):
    tasks=[dict(id='T004',enabled=True,skipped=False),dict(id='T005',enabled=True,skipped=True),dict(id='T006',enabled=False,skipped=False)]
    takes=[]
    for i in range(4):
        raw=storage.save_pcm(tmp_path,f'raw/{i}.f32',np.full((16,1),.1*(i+1),np.float32))
        edit=storage.save_pcm(tmp_path,f'edit/{i}.f32',np.full((20,1),.05*(i+1),np.float32))
        takes.append(dict(id=f'take{i}',task_snapshot=tasks[i] if i<3 else None,config=dict(sample_rate=48000,channels=1,roles=['microphone']),versions=[dict(id=f'raw{i}',kind='raw',spans=[raw]),dict(id=f'gain{i}',kind='gain',spans=[edit])],head=1))
    return dict(tasks=tasks,takes=takes,selected={'T004':'take0','T005':'take1','T006':'take2','__free__':'take3'})


@pytest.mark.parametrize('mode,expected',[
    ('current',['gain0']),('selection',['gain0']),('selected',['gain0']),
    ('all',['gain0','gain1','gain2','gain3']),
    ('versions',['raw0','gain0','raw1','gain1','raw2','gain2','raw3','gain3'])])
def test_export_each_ui_scope_and_real_wav_contents(tmp_path,mode,expected):
    project=fixture_project(tmp_path);before=copy.deepcopy(project)
    items=export_items(project,dict(mode=mode,id='take0',start=3,end=10))
    assert [i['version']['id'] for i in items]==expected
    result=tmp_path/'result.json'
    export_audio(tmp_path,items,tmp_path/'export','FLOAT',result,tmp_path/'cancel')
    receipt=storage.read_json(result)
    assert receipt['state']=='complete' and receipt['success_count']==len(expected)
    for planned,entry in zip(items,receipt['items']):
        path=tmp_path/'export'/receipt['directory_name']/entry['files'][0]['name']
        actual,rate=sf.read(path,dtype='float32',always_2d=True)
        reference=np.concatenate(list(storage.iter_audio(tmp_path,planned['version']['spans'])))
        np.testing.assert_array_equal(actual,reference)
        assert rate==48000 and len(actual)==(7 if mode=='selection' else 16 if entry['version_id'].startswith('raw') else 20)
    assert project==before


def test_raw_scopes_batch_without_current_take_and_invalid_ranges(tmp_path):
    project=fixture_project(tmp_path)
    assert [i['version']['id'] for i in export_items(project,dict(mode='all',raw=True))]==['raw0','raw1','raw2','raw3']
    assert [i['version']['id'] for i in export_items(project,dict(mode='selected',raw=True))]==['raw0']
    assert len(export_items(project,dict(mode='versions',raw=True)))==8
    for body in [dict(mode='invalid'),dict(mode='current',id='missing'),dict(mode='selection',id='take0',raw=True,start=5,end=19),dict(mode='selection',id='take0',start=5,end=5)]:
        with pytest.raises(ValueError):export_items(project,body)


def test_atomic_json_retries_only_transient_windows_errors(tmp_path,monkeypatch):
    path=tmp_path/'process.json';storage.atomic_json(path,{'state':'running'})
    replace=storage.os.replace;calls=[]
    def transient(source,target):
        calls.append(None)
        if len(calls)<3:
            assert storage.read_json(path)=={'state':'running'}
            exc=PermissionError('sharing violation');exc.winerror=5;raise exc
        replace(source,target)
    monkeypatch.setattr(storage.os,'replace',transient)
    monkeypatch.setattr(storage.time,'sleep',lambda _:None)
    storage.atomic_json(path,{'state':'complete'})
    assert len(calls)==3 and storage.read_json(path)=={'state':'complete'}


def test_atomic_json_real_windows_reader_sharing(tmp_path,monkeypatch):
    if os.name!='nt':pytest.skip('Windows sharing flags')
    path=tmp_path/'process.json';storage.atomic_json(path,{'state':'running'})
    reader=path.open('rb');replace=storage.os.replace;blocked=threading.Event();errors=[]
    def observed(source,target):
        try:return replace(source,target)
        except PermissionError:
            blocked.set();raise
    monkeypatch.setattr(storage.os,'replace',observed)
    def writer():
        try:storage.atomic_json(path,{'state':'complete'})
        except Exception as exc:errors.append(exc)
    worker=threading.Thread(target=writer);worker.start()
    try:
        assert blocked.wait(5),'reader did not produce the expected sharing denial'
        assert reader.read()==b'{\n  "state": "running"\n}'
    finally:
        reader.close();worker.join(5)
    assert not worker.is_alive() and not errors
    assert storage.read_json(path)=={'state':'complete'}


def test_atomic_json_permanent_permission_failure_preserves_old_result(tmp_path,monkeypatch):
    path=tmp_path/'process.json';storage.atomic_json(path,{'state':'running'})
    def denied(*_):
        exc=PermissionError('permanent denial');exc.winerror=5;raise exc
    monkeypatch.setattr(storage.os,'replace',denied)
    monkeypatch.setattr(storage.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError):storage.atomic_json(path,{'state':'complete'})
    assert storage.read_json(path)=={'state':'running'}
