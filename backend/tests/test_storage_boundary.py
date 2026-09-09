"""HTTP-only storage spy; does not prove PG transactions or disk persistence."""
from uuid import uuid4
from fastapi.testclient import TestClient
from account_double import MemoryAccountStore
from ptb_api.main import create_app
from ptb_api.auth import AuthSettings
from ptb_api.quota import StorageError, CHUNK_BYTES

ORIGIN='https://storage.test'

class RejectedStorage:
    def __init__(self): self.calls=[]
    def fail(self, owner, *args):
        self.calls.append(owner)
        raise StorageError('asset_not_found',404)
    create=append=finalize=metadata=delete=fail


def login(client):
    token=client.get('/api/v1/auth/challenge').json()['csrf_token']
    response=client.post('/api/v1/auth/login',json={'username':'alice','password':'test-password-only'},headers={'Origin':ORIGIN,'X-CSRF-Token':token})
    assert response.status_code==200
    v=response.json()
    return {'Origin':ORIGIN,'X-CSRF-Token':v['csrf_token'],'X-PTB-Account':v['user']['id']}


def test_storage_write_requires_session_origin_csrf_and_trusted_owner():
    accounts=MemoryAccountStore();accounts.create_user('alice','test-password-only');spy=RejectedStorage()
    app=create_app(account_store=accounts,auth_settings=AuthSettings(origin=ORIGIN,signing_key='test-only-'*8),storage=spy)
    key=str(uuid4());body={'project_id':str(uuid4()),'name':'声调.wav','idempotency_key':'test-key-12345678'}
    with TestClient(app,base_url=ORIGIN) as c:
        assert c.post('/api/v1/uploads',json=body,headers={'Origin':ORIGIN}).status_code==401
        h=login(c)
        assert c.post('/api/v1/uploads',json=body).status_code==403
        assert c.post('/api/v1/uploads',json=body|{'owner_id':'someone'},headers=h).status_code==422
        assert c.delete('/api/v1/assets/'+key).status_code==403
        assert c.delete('/api/v1/assets/'+key,headers=h|{'X-PTB-Account':'stale'}).status_code==409
        assert c.get('/api/v1/assets/'+key,headers={'Host':'evil.test'}).status_code==403
        assert not spy.calls
        assert c.post('/api/v1/uploads',json=body,headers=h).status_code==404
        assert spy.calls==[h['X-PTB-Account']]
        assert c.delete('/api/v1/assets/'+key,headers=h).status_code==404
        assert c.get('/api/v1/assets/'+key+'/content?expected_account='+str(uuid4()),headers=h).status_code==409


def test_binary_blocks_have_separate_bounded_limit_and_no_temp_disk():
    accounts=MemoryAccountStore();accounts.create_user('alice','test-password-only');spy=RejectedStorage()
    app=create_app(account_store=accounts,auth_settings=AuthSettings(origin=ORIGIN,signing_key='test-only-'*8),storage=spy)
    with TestClient(app,base_url=ORIGIN) as c:
        h=login(c)|{'Content-Type':'application/octet-stream'};path='/api/v1/uploads/'+str(uuid4())+'/blocks?offset=0'
        assert c.put(path,content=b'a'*CHUNK_BYTES,headers=h).status_code==404
        assert len(spy.calls)==1
        assert c.put(path,content=b'a'*(CHUNK_BYTES+1),headers=h).status_code==413
        assert len(spy.calls)==1
        assert c.post('/api/v1/uploads',content=b'a'*17000,headers=h).status_code==413
        response=c.put(path,content=b'x',headers=h|{'Origin':'https://evil.test'})
        assert response.status_code==403 and response.headers['cache-control']=='no-store'


def test_web_storage_never_applies_server_retention_to_desktop():
    for mode,status in [('local',404),('server',503)]:
        with TestClient(create_app(mode)) as c:
            assert c.get('/api/v1/capabilities').json()['storage_operations']==[]
            assert c.get('/api/v1/storage/usage').status_code==status
            assert c.get('/api/v1/assets?project_id='+str(uuid4())).status_code==status


def test_download_rechecks_owner_and_stops_before_an_expired_next_block():
    import asyncio
    import pytest
    from ptb_api.assets import PrivateDownload
    sent=[]
    class Context:
        calls=0
        def session(self, request):
            self.calls+=1
            return {'id':'owner'}
    class FileSpy:
        calls=0
        def read_block(self, owner, asset_id, offset, size):
            assert owner=='owner'
            self.calls+=1
            if self.calls==2: raise StorageError('asset_expired',410)
            return b'a'*size
    ctx=Context();store=FileSpy()
    response=PrivateDownload(store,ctx,None,'owner',{'id':'file','name':'音频.wav','size_bytes':CHUNK_BYTES*2},0,CHUNK_BYTES*2,False)
    async def run():
        async def send(message): sent.append(message)
        with pytest.raises(StorageError,match='asset_expired'):
            await response({'type':'http'},None,send)
    asyncio.run(run())
    assert ctx.calls==2 and store.calls==2
    bodies=[m for m in sent if m['type']=='http.response.body']
    assert len(bodies)==1 and len(bodies[0]['body'])==CHUNK_BYTES and bodies[0]['more_body']
