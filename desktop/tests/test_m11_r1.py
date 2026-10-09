"""M11-R1 explicit source selection through native grants, no real user files."""
from types import SimpleNamespace
import pytest
from ptb_desktop.m11_bridge import M11Bridge


class ImportService:
    def __init__(self):self.names=[]
    def import_input(self,raw,name,role):
        self.names.append(name)
        return dict(asset_id=name,sha256='a'*64)


@pytest.mark.parametrize('source,expected',[('.lab','a.lab'),('.TextGrid','a.TextGrid'),('.txt','a.txt')])
def test_m11_r1_reads_only_selected_transcript(tmp_path,source,expected):
    for name in ('a.wav','a.lab','a.TextGrid','a.txt'):(tmp_path/name).write_bytes(b'fixture')
    service=ImportService();bridge=M11Bridge(SimpleNamespace(service=service))
    grant=bridge.grant('corpus',tmp_path)
    value=bridge.invoke(dict(op='m11_corpus',id=grant['id'],transcript_source=source))
    assert service.names==['a.wav',expected]
    assert value[0]['transcript_format']==source
    assert len(list(tmp_path.iterdir()))==4


def test_m11_r1_ambiguity_and_missing_source_do_not_import_partial_assets(tmp_path):
    for name in ('a.wav','a.lab','a.TextGrid','b.wav','b.lab'):(tmp_path/name).write_bytes(b'fixture')
    service=ImportService();bridge=M11Bridge(SimpleNamespace(service=service))
    grant=bridge.grant('corpus',tmp_path)
    for source,match in [('auto','转写来源'),('.TextGrid','b.wav'),('invalid','不受支持')]:
        with pytest.raises(Exception,match=match):bridge.invoke(dict(op='m11_corpus',id=grant['id'],transcript_source=source))
        assert service.names==[]


def test_m11_r2_dictionary_grant_reads_only_authorized_bounded_file(tmp_path):
    dictionary=tmp_path/'拼音.dict';dictionary.write_text('a1\ta1\n',encoding='utf8')
    service=ImportService();bridge=M11Bridge(SimpleNamespace(service=service))
    grant=bridge.grant('dictionary',dictionary)
    value=bridge.invoke(dict(op='m11_dictionary',id=grant['id']))
    assert service.names==['拼音.dict'] and value['asset_id']=='拼音.dict'
    wrong=bridge.grant('model',dictionary)
    with pytest.raises(Exception,match='对应资源'):
        bridge.invoke(dict(op='m11_dictionary',id=wrong['id']))
    dictionary.write_bytes(b'')
    with pytest.raises(Exception,match='为空'):
        bridge.invoke(dict(op='m11_dictionary',id=grant['id']))


def test_m11_r2_check_can_reuse_registered_environment_and_resources(tmp_path,monkeypatch):
    import ptb_worker.mfa.runtime as runtime
    env=tmp_path/'env';env.mkdir()
    model=tmp_path/'model.zip';model.write_bytes(b'model')
    dictionary=tmp_path/'dictionary.dict';dictionary.write_text('a1\ta1\n')
    import hashlib
    records=dict(runtimes=[dict(id='r',path=str(env),validated=True)],models=[dict(id='m',model=str(model),dictionary=str(dictionary),validated_runtime='r',model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),dictionary_sha256=hashlib.sha256(dictionary.read_bytes()).hexdigest())])
    monkeypatch.setattr(runtime,'load_registry',lambda:records)
    class ComponentService:
        def request(self,path,method,body):
            assert path=='/api/v1/jobs/m11/component' and method=='POST'
            return body
    bridge=M11Bridge(SimpleNamespace(service=ComponentService()))
    value=bridge.invoke(dict(op='m11_component',body=dict(action='check',runtime_id='r',model_id='m')))
    assert value==dict(action='check',runtime=str(env),model=str(model),dictionary=str(dictionary))
    override=tmp_path/'other.dict';override.write_text('a1\ta1\n')
    grant=bridge.grant('dictionary',override)
    value=bridge.invoke(dict(op='m11_component',body=dict(action='check',runtime_id='r',model_id='m',dictionary=grant['id'])))
    assert value['dictionary']==str(override)
    for bad in ('missing',''):
        with pytest.raises(Exception,match='m11_component_not_registered'):
            bridge.invoke(dict(op='m11_component',body=dict(action='check',runtime_id=bad,model_id='m')))
    records['runtimes'][0]['validated']=False
    with pytest.raises(Exception,match='m11_self_test_required'):
        bridge.invoke(dict(op='m11_component',body=dict(action='check',runtime_id='r',model_id='m')))
    records['runtimes'][0]['validated']=True
    dictionary.write_bytes(b'changed')
    with pytest.raises(Exception,match='m11_model_changed'):
        bridge.invoke(dict(op='m11_component',body=dict(action='check',runtime_id='r',model_id='m')))
