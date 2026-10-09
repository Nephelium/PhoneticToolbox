"""M08-R1 native file handling; synthetic bytes, isolated directories, no DB."""
import base64
import hashlib
from types import SimpleNamespace
from uuid import uuid4
import pytest
from ptb_desktop.file_provider import FileProvider, FileAccessError
from ptb_desktop.m08_bridge import M08Bridge
from ptb_desktop.task_bridge import PROJECT


@pytest.fixture
def native(tmp_path):
    destination=tmp_path/'saved';destination.mkdir()
    provider=FileProvider();grant=provider.choose('output',lambda:str(destination))
    raw=b'RIFF synthetic M08 result';result_id=str(uuid4());job_id=str(uuid4())
    item=dict(id=result_id,name='合成_modified_1.wav',sha256=hashlib.sha256(raw).hexdigest())
    job=dict(id=job_id,operation='pitch_manipulation',state='succeeded',result_manifest=dict(files=[item]))
    calls=[]
    def request(url,method,body):
        calls.append((url,body))
        return dict(removed=body['ids'],failed=[]) if url.endswith('/remove') else dict(renamed=body['ids'])
    service=SimpleNamespace(get=lambda _:job,request=request)
    host=SimpleNamespace(provider=provider,service=service,invoke=lambda _:dict(base64=base64.b64encode(raw).decode()),record_export=lambda ids:None)
    bridge=M08Bridge(host)
    body=dict(op='m08_export',job=job_id,id=result_id,directory=grant['id'],direct=True)
    return bridge,body,destination,raw,calls


def test_receipt_only_after_successful_verified_export(native):
    bridge,body,directory,raw,_=native;receipts=[]
    bridge.bridge.record_export=lambda ids:receipts.append(ids)
    bridge.export(body)
    assert receipts==[[body['id']]]
    assert (directory/'合成_modified_1.wav').read_bytes()==raw
    bridge.export(body)
    assert receipts==[[body['id']],[body['id']]]
    (directory/'合成_modified_1.wav').write_bytes(b'changed outside')
    with pytest.raises(FileAccessError):bridge.export(body)
    assert len(receipts)==2


def test_direct_export_collision_and_idempotent_retry(native):
    bridge,body,directory,raw,_=native
    occupied=directory/'合成_modified_1.wav';occupied.write_bytes(b'keep existing')
    first=bridge.export(body);assert first['name']=='合成_modified_1 (2).wav'
    assert first['directory']==str(directory.resolve())
    assert (directory/first['name']).read_bytes()==raw
    assert bridge.export(body)==first
    assert len(list(directory.iterdir()))==2 and occupied.read_bytes()==b'keep existing'


def test_retry_refuses_modified_export_and_rechecks_expiry(native):
    bridge,body,directory,_,_=native
    first=bridge.export(body);(directory/first['name']).write_bytes(b'external edit')
    with pytest.raises(FileAccessError,match='外部修改'):bridge.export(body)
    def expired(_):raise FileAccessError('result expired')
    bridge.bridge.invoke=expired
    with pytest.raises(FileAccessError,match='expired'):bridge.export(body)


def test_two_destinations_remain_explicitly_managed(native,tmp_path):
    bridge,body,directory,raw,_=native
    first=bridge.export(body);other=tmp_path/'second';other.mkdir()
    grant=bridge.provider.choose('output',lambda:str(other));bridge.export(dict(body,directory=grant['id']))
    scope=dict(project_id=PROJECT,ids=[body['id']],names=['renamed.wav'])
    bridge.invoke(dict(op='m08_call',action='rename',body=scope))
    assert (directory/'renamed.wav').read_bytes()==(other/'renamed.wav').read_bytes()==raw
    bridge.invoke(dict(op='m08_call',action='remove',body=scope))
    assert not list(directory.iterdir()) and not list(other.iterdir())
    assert not bridge.exports[body['id']]


def test_rename_conflict_stops_before_managed_mutation(native):
    bridge,body,directory,_,calls=native;bridge.export(body)
    (directory/'occupied.wav').write_bytes(b'keep')
    with pytest.raises(FileAccessError,match='冲突'):
        bridge.invoke(dict(op='m08_call',action='rename',body=dict(project_id=PROJECT,ids=[body['id']],names=['occupied.wav'])))
    assert not calls and (directory/'occupied.wav').read_bytes()==b'keep'


def test_legacy_numbering_and_input_grant_rejection(native):
    bridge,body,directory,raw,_=native
    (directory/'合成_modified_9.wav').write_bytes(b'keep')
    assert bridge.export(dict(body,direct=False))['name']=='合成_modified_10.wav'
    grant=bridge.provider.choose('input',lambda:str(directory))
    with pytest.raises(FileAccessError,match='输出目录'):bridge.export(dict(body,directory=grant['id']))


def test_failed_write_has_no_success_record_and_retry_works(native,monkeypatch):
    import ptb_desktop.m08_bridge as module
    bridge,body,directory,raw,_=native
    with monkeypatch.context() as patch:
        patch.setattr(module.os,'fsync',lambda _:(_ for _ in ()).throw(OSError('disk full')))
        with pytest.raises(OSError,match='disk full'):bridge.export(body)
    assert not list(directory.iterdir()) and not bridge.exports
    saved=bridge.export(body);assert (directory/saved['name']).read_bytes()==raw
