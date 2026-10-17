from types import SimpleNamespace
import pytest
from ptb_worker import preview_cache as module


@pytest.fixture(autouse=True)
def host_profile_fixture(monkeypatch,tmp_path):
    from ptb_worker.native import linux_runtime
    profile=tmp_path/'profile.json';profile.write_text('{}',encoding='utf8')
    monkeypatch.setattr(linux_runtime,'load_profile',lambda:(profile,{}))


def test_bounds_ttl_and_no_mutable_alias():
    now=[0.];cache=module.PreviewCache(max_bytes=30,max_entries=2,ttl=3,clock=lambda:now[0])
    cache.put('a',{'x':[1]});value=cache.get('a');value['x'][0]=9;assert cache.get('a')=={'x':[1]}
    cache.put('b',{'x':[2]});cache.put('c',{'x':[3]});assert cache.get('a') is None and cache.bytes<=30
    cache.put('oversized',{'x':'x'*100});assert cache.get('oversized') is None
    now[0]=3;assert cache.get('c') is None and cache.bytes==0


def test_owner_source_config_and_failures(monkeypatch):
    monkeypatch.setattr(module,'cache',module.PreviewCache());calls=[]
    @module.memoized_preview('test')
    def render(raw,config):
        calls.append((raw,config))
        if config.get('fail'):raise ValueError('failed')
        return {'result':[len(calls)]}
    with module.preview_scope(('a','file')):
        assert render(b'a',{'channel':0})==render(b'a',{'channel':0})=={'result':[1]}
        render(b'a',{'channel':1});render(b'b',{'channel':0})
        for _ in range(2):
            with pytest.raises(ValueError):render(b'a',{'fail':True})
    with module.preview_scope(('b','file')):render(b'a',{'channel':0})
    assert len(calls)==6


def test_cache_hit_still_rechecks_asset_access(monkeypatch):
    from fastapi import FastAPI,HTTPException
    from fastapi.testclient import TestClient
    from ptb_api.assets import create_storage_router
    from ptb_worker import parameter_preview
    import hashlib
    monkeypatch.setattr(module,'cache',module.PreviewCache());state={'expired':False,'calls':0};raw=b'synthetic-table'
    @module.memoized_preview('table-test')
    def render(raw,name):
        state['calls']+=1
        return dict(sha256=hashlib.sha256(raw).hexdigest(),columns=['Time_s'],kinds=['number'],rows=[[0.]])
    monkeypatch.setattr(parameter_preview,'render',render)
    def metadata(owner,asset):
        if state['expired']:raise HTTPException(410,'asset_expired')
        return dict(name='result.ptb.sqlite',size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    ctx=SimpleNamespace(available=lambda:None,session=lambda request:{'id':'a'},mutation=lambda request:{'id':'a'})
    store=SimpleNamespace(metadata=metadata,read_block=lambda owner,asset,offset,size:raw[offset:offset+size])
    app=FastAPI();app.include_router(create_storage_router(ctx,store))
    with TestClient(app) as client:
        url='/api/v1/assets/00000000-0000-4000-8000-000000000001/parameters'
        assert client.get(url).status_code==200;assert client.get(url).status_code==200;assert state['calls']==1
        state['expired']=True;assert client.get(url).status_code==410;assert state['calls']==1
