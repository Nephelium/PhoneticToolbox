"""Actual Windows-owned Praat process and local/server preview boundaries."""
from pathlib import Path
from uuid import uuid4
import hashlib
import pytest
from fastapi.testclient import TestClient
from ptb_api.main import create_app
from ptb_api.auth import AuthSettings
from ptb_worker.spectrogram_preview import render,preview_slot,PreviewError
from account_double import MemoryAccountStore
from m01_preview_double import PreviewStorage

ROOT=Path(__file__).resolve().parents[2]
WAV=(ROOT/'frontend/src/assets/SYN-EGG-44100.wav').read_bytes()


def test_actual_child_returns_bounded_pixels_and_timeout_reaps_owned_process(monkeypatch):
    import ptb_worker.spectrogram_preview as preview
    import subprocess
    original=subprocess.Popen;owned=[]
    def capture(*args,**kwargs):
        process=original(*args,**kwargs);owned.append(process);return process
    monkeypatch.setattr(preview.subprocess,'Popen',capture)
    value=render(WAV,channel=0,start=0,end=.8,width=300)
    assert value['backend']=='praat' and value['width']<=302
    assert value['parselmouth_version']=='0.4.7' and len(value['pixels_base64'])<400000
    with pytest.raises(PreviewError,match='preview_timeout'):render(WAV,channel=0,start=0,end=.8,width=300,timeout=.0001)
    assert len(owned)==2 and all(p.poll() is not None for p in owned)


def test_local_preview_requires_session_and_returns_actual_audio_hash():
    token='local-test-only-'+('x'*32);origin='http://127.0.0.1:12345'
    app=create_app('local',local_token=token,local_origin=origin)
    path='/api/v1/preview/spectrogram?channel=0&start=0&end=0.8&width=300'
    headers={'Origin':origin,'Authorization':'Bearer '+token,'Content-Type':'application/octet-stream'}
    with TestClient(app) as c:
        assert c.post(path,content=WAV).status_code==403
        with preview_slot():assert c.post(path,content=WAV,headers=headers).status_code==429
        response=c.post(path,content=WAV,headers=headers)
        assert response.status_code==200,response.text
        assert response.json()['sha256']==hashlib.sha256(WAV).hexdigest()
        assert response.headers['cache-control']=='no-store'
        assert c.post(path,content=b'bad',headers=headers).status_code==422


def test_server_preview_owner_and_expiry_are_rechecked_after_computation(monkeypatch):
    import ptb_worker.spectrogram_preview as preview
    accounts=MemoryAccountStore();accounts.create_user('alice','test-only');owner=accounts.users['alice']['id']
    store=PreviewStorage();key=store.add(owner,str(uuid4()),'sound.wav',WAV)
    foreign=store.add(str(uuid4()),str(uuid4()),'private.wav',WAV)
    origin='https://preview.test';app=create_app(account_store=accounts,auth_settings=AuthSettings(origin=origin,signing_key='test-only'*8),storage=store)
    suffix='/spectrogram?channel=0&start=0&end=0.8&width=300'
    with TestClient(app,base_url=origin) as c:
        assert c.get('/api/v1/assets/'+key+suffix).status_code==401
        token=c.get('/api/v1/auth/challenge').json()['csrf_token']
        assert c.post('/api/v1/auth/login',json={'username':'alice','password':'test-only'},headers={'Origin':origin,'X-CSRF-Token':token}).status_code==200
        assert c.get('/api/v1/assets/'+foreign+suffix).status_code==404
        original=preview.render
        def expire(*args,**kwargs):
            result=original(*args,**kwargs);store.assets[key]['expires_at']=0;return result
        monkeypatch.setattr(preview,'render',expire)
        assert c.get('/api/v1/assets/'+key+suffix).status_code==410
