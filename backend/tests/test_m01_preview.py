"""M01-E trusted HTTP boundary and parser parity; store is explicitly a test double."""
from fastapi.testclient import TestClient
from account_double import MemoryAccountStore
from m01_preview_double import PreviewStorage,GRID
from ptb_api.main import create_app
from ptb_api.auth import AuthSettings
from pathlib import Path
from uuid import uuid4
import pytest


def test_owned_textgrid_preview_has_actual_hash_and_rejects_foreign_and_expired():
    accounts=MemoryAccountStore();accounts.create_user('alice','preview-test-only');owner=accounts.users['alice']['id']
    store=PreviewStorage();key=store.add(owner,str(uuid4()),'labels.TextGrid',GRID)
    foreign=store.add(str(uuid4()),str(uuid4()),'foreign.TextGrid',GRID)
    app=create_app(account_store=accounts,auth_settings=AuthSettings(origin='https://preview.test',signing_key='test-only-key'*4),storage=store)
    with TestClient(app,base_url='https://preview.test') as c:
        assert c.get('/api/v1/assets/'+key+'/textgrid').status_code==401
        token=c.get('/api/v1/auth/challenge').json()['csrf_token']
        assert c.post('/api/v1/auth/login',json={'username':'alice','password':'preview-test-only'},headers={'Origin':'https://preview.test','X-CSRF-Token':token}).status_code==200
        data=c.get('/api/v1/assets/'+key+'/textgrid',headers={'X-PTB-Account':owner})
        assert data.status_code==200
        assert data.json()['sha256']==store.assets[key]['sha256']
        assert data.json()['tiers'][0]['intervals']==[{'xmin':0.,'xmax':.4,'text':'a'},{'xmin':.4,'xmax':.8,'text':'i'}]
        assert data.headers['cache-control']=='no-store'
        negative=GRID.replace(b'\n0\n',b'\n-1\n')
        negative_id=store.add(owner,str(uuid4()),'negative.TextGrid',negative)
        negative_result=c.get('/api/v1/assets/'+negative_id+'/textgrid')
        assert negative_result.status_code==200
        assert negative_result.json()['tiers'][0]['intervals'][0]['xmin']==-1.
        invalid=store.add(owner,str(uuid4()),'broken.TextGrid',b'invalid')
        assert c.get('/api/v1/assets/'+invalid+'/textgrid').status_code==422
        too_large=store.add(owner,str(uuid4()),'large.TextGrid',GRID)
        store.assets[too_large]['size_bytes']=2_000_001
        assert c.get('/api/v1/assets/'+too_large+'/textgrid').status_code==422
        store.assets[key]['sha256']='0'*64
        assert c.get('/api/v1/assets/'+key+'/textgrid').status_code==409
        assert c.get('/api/v1/assets/'+foreign+'/textgrid').status_code==404
        assert c.get('/api/v1/assets/'+key+'/textgrid',headers={'X-PTB-Account':str(uuid4())}).status_code==409
        store.assets[key]['expires_at']=0.
        assert c.get('/api/v1/assets/'+key+'/textgrid').status_code==410


def test_preview_parser_import_is_lightweight():
    import subprocess,sys
    subprocess.run([sys.executable,'-c','import sys; from phonetic_core.textgrid import parse_textgrid; assert "numpy" not in sys.modules; assert "phonetic_core.acoustic" not in sys.modules'],check=True)
