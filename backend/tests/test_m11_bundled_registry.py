import json
import pytest
from ptb_worker.mfa.runtime import load_registry

def fixture(tmp_path,monkeypatch):
    built=tmp_path/'built';built.mkdir();(built/'runtime').mkdir();(built/'receipt.json').write_text('{}')
    (built/'model.zip').write_bytes(b'model');(built/'dictionary.dict').write_text('a1 a')
    registry=dict(schema='m11-registry/1',runtimes=[dict(id='builtin',path='runtime',receipt='receipt.json')],
                  models=[dict(id='model',model='model.zip',dictionary='dictionary.dict')])
    (built/'registry-bundled.json').write_text(json.dumps(registry))
    custom=tmp_path/'custom';monkeypatch.setenv('PTB_M11_COMPONENT_ROOT',str(custom))
    monkeypatch.setenv('PTB_M11_BUNDLED_COMPONENTS',str(built))
    return built,custom,registry

def test_builtin_relocates_without_writing_user_registry(tmp_path,monkeypatch):
    built,custom,_=fixture(tmp_path,monkeypatch);result=load_registry()
    assert result['runtimes'][0]['path']==str(built/'runtime')
    assert result['models'][0]['dictionary']==str(built/'dictionary.dict')
    assert not custom.exists()
    moved=tmp_path/'new-version';built.rename(moved);monkeypatch.setenv('PTB_M11_BUNDLED_COMPONENTS',str(moved))
    assert load_registry()['runtimes'][0]['path']==str(moved/'runtime')

def test_user_import_wins_and_is_not_rewritten(tmp_path,monkeypatch):
    _,custom,_=fixture(tmp_path,monkeypatch);custom.mkdir()
    state=dict(schema='m11-registry/1',runtimes=[dict(id='builtin',path='user-selected')],models=[])
    path=custom/'registry.json';path.write_text(json.dumps(state));before=path.read_bytes()
    assert load_registry()['runtimes']==state['runtimes']
    assert path.read_bytes()==before

@pytest.mark.parametrize('name',['../outside','C:/absolute','runtime/../escape'])
def test_builtin_path_escape_rejected(tmp_path,monkeypatch,name):
    built,_,registry=fixture(tmp_path,monkeypatch);registry['runtimes'][0]['path']=name
    (built/'registry-bundled.json').write_text(json.dumps(registry))
    with pytest.raises(ValueError):load_registry()
