"""M01-R2 legacy normalization at the desktop input boundary; no source writes."""
import hashlib
import pickle
import pytest
from ptb_desktop.file_provider import FileProvider, FileAccessError
from ptb_desktop.task_bridge import TaskBridge
from ptb_worker.io.lip import decode_lip, encode_lip
from ptb_worker.legacy_conversion import convert


class InputService:
    def __init__(self):
        self.imports=[]
        self.body=None
        self.conversions=0
        self.after_conversion=lambda:None

    def binary(self,path,method,payload,**kwargs):
        assert path=='/api/v1/jobs/local-lip-conversion' and method=='POST'
        self.conversions+=1
        result=convert(payload)
        self.after_conversion()
        return result

    def import_input(self,raw,name,role):
        self.imports.append((raw,name,role))
        return {'asset_id':str(len(self.imports)),'sha256':hashlib.sha256(raw).hexdigest()}

    def request(self,path,method,body):
        assert path=='/api/v1/jobs/batches/create' and method=='POST'
        self.body=body
        return {'id':'test-batch'}


def inputs(tmp_path):
    (tmp_path/'a.wav').write_bytes(b'unchanged audio fixture')
    (tmp_path/'a.pkl').write_bytes(pickle.dumps({'absolute_timestamps':[100.,100.1],'open':[1.,2.],
        'metadata':{'lip_manual_offset':.02}}))
    (tmp_path/'a_timestamps.pkl').write_bytes(pickle.dumps({'start_time':99.}))
    (tmp_path/'b.lip.json').write_bytes(encode_lip({'relative_times':[0.,.1],'open':[3.,4.]}))
    provider=FileProvider();grant=provider.choose('input',lambda:tmp_path)
    entries={f['name']:f['id'] for f in provider.list(grant['id'])}
    return provider,entries


def test_mixed_json_and_pickle_submit_normalizes_only_legacy_without_source_outputs(tmp_path):
    provider,entries=inputs(tmp_path);service=InputService();bridge=TaskBridge(provider,service)
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    bridge.invoke({'op':'submit','operation':'acoustic_analysis','inputs':[
        {'audio':entries['a.wav'],'lip':entries['a.pkl']},
        {'audio':entries['a.wav'],'lip':entries['b.lip.json']}],
        'config':{},'layer':None,'idempotency_key':'test-key'})
    lips=[(raw,name) for raw,name,role in service.imports if role=='lip']
    assert [name for _,name in lips]==['a.lip.json','b.lip.json']
    old=decode_lip(lips[0][0]);assert old['open']==[1.,2.]
    assert old['metadata']['audio_first_frame_time']==99.
    assert old['metadata']['lip_manual_offset']==.02
    assert lips[1][0]==before['b.lip.json'] and service.conversions==1
    assert len(service.body['inputs'])==2
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir()}==before


def test_pickle_anchor_has_priority_over_companion(tmp_path):
    provider,entries=inputs(tmp_path)
    (tmp_path/'a.pkl').write_bytes(pickle.dumps({'absolute_timestamps':[100.,100.1],'area':[4.,5.],
        'metadata':{'audio_first_frame_time':98.}}))
    grant=next(iter(provider.directories));entries={f['name']:f['id'] for f in provider.list(grant)}
    raw,_,found=TaskBridge(provider,InputService()).lip_input(entries['a.pkl'])
    assert found and decode_lip(raw)['metadata']['audio_first_frame_time']==98.


@pytest.mark.parametrize('changed',['a.pkl','a_timestamps.pkl'])
def test_legacy_input_changed_during_conversion_is_not_imported(tmp_path,changed):
    provider,entries=inputs(tmp_path);service=InputService()
    service.after_conversion=lambda:(tmp_path/changed).write_bytes(b'changed fixture')
    bridge=TaskBridge(provider,service)
    with pytest.raises(FileAccessError):bridge.lip_input(entries['a.pkl'])
    assert not service.imports and not (tmp_path/'a.lip.json').exists()


def test_timestamps_cannot_be_selected_as_lip_data(tmp_path):
    provider,entries=inputs(tmp_path);service=InputService()
    with pytest.raises(FileAccessError):TaskBridge(provider,service).lip_input(entries['a_timestamps.pkl'])
    assert service.conversions==0
