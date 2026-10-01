"""Transient-session boundaries using the explicitly authorized real recording."""
import os
from pathlib import Path
import time
import pytest
from ptb_worker.egg_interactive import InteractivePreview
from ptb_worker.spectrogram_preview import PreviewError

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(r'C:\Users\13680\Desktop\project\音频数据\EGG测试\3.wav')
CONFIG=dict(mode='preview',roi_start=40,roi_end=40.5,micro_center=40.25,keep_gci_f0=False,keep_praat_f0=False)


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setenv('PTB_EGG_PYTHON',str(ROOT/'.venv/m03-compatible/python.exe'))
    value=InteractivePreview()
    yield value
    value.close()


def test_reuses_process_checks_owner_and_releases(manager):
    opened=manager.open(('owner','asset'),SOURCE.read_bytes());process=manager.process
    one=manager.update(('owner','asset'),opened['session_id'],CONFIG)
    assert one['audio_base64'] and one['input_sha256']==opened['sha256']
    with pytest.raises(PreviewError,match='egg_preview_expired'):
        manager.update(('other','asset'),opened['session_id'],CONFIG)
    manager.release(('other','asset'),opened['session_id'])
    two=manager.update(('owner','asset'),opened['session_id'],{**CONFIG,'micro_center':40.3})
    assert not two['audio_base64'] and two['preview']['micro_center']==40.3
    assert manager.process is process and process.poll() is None
    manager.release(('owner','asset'),opened['session_id'])
    assert process.poll() is not None and manager.process is None


def test_error_does_not_poison_current_source(manager):
    opened=manager.open('owner',SOURCE.read_bytes())
    with pytest.raises(PreviewError):
        manager.update('owner',opened['session_id'],{**CONFIG,'roi_end':100})
    value=manager.update('owner',opened['session_id'],CONFIG)
    assert value['preview']['micro_center']==40.25


def test_deadline_reaps_owned_process(manager):
    manager.open('owner',SOURCE.read_bytes());process=manager.process
    with manager.lock,pytest.raises(PreviewError,match='preview_timeout'):
        manager._rpc(dict(config=CONFIG),timeout=.00001)
    assert process.poll() is not None and manager.process is None


def test_idle_expiry_releases_memory(manager):
    manager.idle_seconds=.1
    manager.open('owner',SOURCE.read_bytes());process=manager.process
    deadline=time.monotonic()+5
    while process.poll() is None and time.monotonic()<deadline: time.sleep(.05)
    assert process.poll() is not None


def test_local_preview_authentication_precedes_science():
    from fastapi.testclient import TestClient
    from ptb_api.main import create_app
    with TestClient(create_app('local',local_token='test-token',local_origin='http://local.test')) as client:
        for path in ['/api/v1/preview/egg','/api/v1/preview/egg/00000000-0000-4000-8000-000000000001']:
            response=client.post(path,json=CONFIG)
            assert response.status_code==403


def test_real_http_response_and_close(monkeypatch):
    from fastapi.testclient import TestClient
    from ptb_api.main import create_app
    monkeypatch.setenv('PTB_EGG_PYTHON',str(ROOT/'.venv/m03-compatible/python.exe'))
    headers={'Origin':'http://local.test','Authorization':'Bearer test-token','Content-Type':'application/octet-stream'}
    with TestClient(create_app('local',local_token='test-token',local_origin='http://local.test')) as client:
        opened=client.post('/api/v1/preview/egg',content=SOURCE.read_bytes(),headers=headers)
        assert opened.status_code==200
        headers['Content-Type']='application/json'
        path='/api/v1/preview/egg/'+opened.json()['session_id']
        response=client.post(path,json=CONFIG,headers=headers)
        assert response.status_code==200
        assert response.json()['preview']['spectral_extent'][0]==40
        assert client.delete(path,headers=headers).status_code==200
        assert client.post(path,json=CONFIG,headers=headers).status_code==410
        assert client.post(path,json={'padding':'x'*17000},headers=headers).status_code==413


def test_asset_ticket_owner_and_expiry_are_rechecked(manager):
    """Controlled account/storage boundary, real science; no database setup."""
    import hashlib
    from fastapi import FastAPI,HTTPException
    from fastapi.testclient import TestClient
    from starlette.responses import JSONResponse
    from ptb_api.assets import create_storage_router
    raw=SOURCE.read_bytes()
    class Context:
        def available(self): pass
        def session(self,request): return {'id':request.headers['x-test-owner']}
        mutation=session
    class Store:
        expired=False
        def metadata(self,owner,asset):
            if self.expired: raise HTTPException(410,'asset_expired')
            return dict(name='3.wav',size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        def read_block(self,owner,asset,offset,size): return raw[offset:offset+size]
    store=Store();app=FastAPI();app.include_router(create_storage_router(Context(),store,egg_preview=manager))
    @app.exception_handler(PreviewError)
    async def error(request,exc): return JSONResponse({'detail':exc.code},status_code=exc.status)
    with TestClient(app) as client:
        path='/api/v1/assets/00000000-0000-4000-8000-000000000001/egg-preview'
        session=client.post(path,headers={'x-test-owner':'one'}).json()['session_id']
        path+='/'+session
        assert client.post(path,headers={'x-test-owner':'two'},json=CONFIG).status_code==410
        assert client.post(path,headers={'x-test-owner':'one'},json=CONFIG).status_code==200
        store.expired=True
        assert client.post(path,headers={'x-test-owner':'one'},json=CONFIG).status_code==410
