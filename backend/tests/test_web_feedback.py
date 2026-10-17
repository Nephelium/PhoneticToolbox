from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ptb_api.auth import AccountContext,AuthSettings
from ptb_api.web_feedback import create_web_feedback_router
from ptb_api.feedback import FeedbackService,create_feedback_app

ORIGIN='https://account.test'
def client(email='owned@example.test'):
    calls=[]
    store=SimpleNamespace(session=lambda _:dict(id='owner',csrf_token='csrf'),verified_email=lambda owner:email)
    ctx=AccountContext(store,AuthSettings(ORIGIN,'k'*32),'server')
    def call(*args):calls.append(args);return {'connected':True,'email':args[0]}
    app=FastAPI();app.include_router(create_web_feedback_router(ctx,SimpleNamespace(call=call)))
    api=TestClient(app,base_url=ORIGIN);api.cookies.set('__Host-ptb-session','s'*32)
    api.headers.update({'Origin':ORIGIN,'X-CSRF-Token':'csrf','X-PTB-Account':'owner'})
    return api,calls,store

def test_current_account_email_is_authoritative_and_cookie_is_required():
    api,calls,store=client()
    assert api.post('/api/v1/web-feedback',json={'operation':'status'}).json()['email']=='owned@example.test'
    assert calls==[('owned@example.test','status',{})]
    for body in ({'operation':'status','email':'other@example.test'},{'operation':'messages','args':{'owner':'other'}},{'operation':'verifyStart','args':{'email':'other@example.test'}}):
        assert api.post('/api/v1/web-feedback',json=body).status_code==422
    assert len(calls)==1
    assert api.post('/api/v1/web-feedback',json={'operation':'status'},headers={'X-PTB-Account':'other'}).status_code==409
    assert api.post('/api/v1/web-feedback',json={'operation':'status'},headers={'X-CSRF-Token':'wrong'}).status_code==403
    assert api.post('/api/v1/web-feedback',json={'operation':'status'},headers={'Origin':'https://other.test'}).status_code==403
    store.session=lambda _:None
    assert api.post('/api/v1/web-feedback',json={'operation':'status'}).status_code==401

def test_unverified_account_does_not_create_feedback_identity():
    api,calls,_=client(None)
    assert api.post('/api/v1/web-feedback',json={'operation':'status'}).status_code==403
    assert calls==[]

def test_internal_verified_identity_reuses_verified_owner_without_claiming_unverified_content(tmp_path):
    service=FeedbackService(tmp_path/'feedback.sqlite',secret='s'*32,admin_token='a'*32,allow_unverified=True)
    service.web_operations=SimpleNamespace(key='k'*32)
    old=service.open_device('owned@example.test','old-device-token')
    service.send(old['owner'],'user','unverified original',[],'original-key')
    with TestClient(create_feedback_app(service),client=('127.0.0.1',50000)) as api:
        url='/api/v1/feedback/internal/web-operation';body={'email':'owned@example.test','operation':'status'}
        assert api.post(url,json=body).status_code==401
        headers={'Authorization':'Bearer '+'k'*32}
        linked=api.post(url,json=body,headers=headers).json()
        assert linked['verified'] and linked['owner']!=old['owner'] and 'token' not in linked
        assert api.post(url,json=body,headers=headers).json()['owner']==linked['owner']
        history=api.post(url,json={**body,'operation':'messages'},headers=headers).json()
        assert history['messages']==[]
        sent=api.post(url,json={**body,'operation':'send','args':{'text':'owned feedback','images':[],'requestId':'synthetic-request'}},headers=headers)
        assert sent.status_code==200
        result=api.post(url,json={**body,'operation':'messages'},headers=headers).json()
        assert result['messages'][0]['text']=='owned feedback' and result['messages'][0]['client']['source']=='web'
        assert service.messages(old['owner'])['messages'][0]['text']=='unverified original'
