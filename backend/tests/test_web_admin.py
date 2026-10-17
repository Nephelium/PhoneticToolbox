from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from ptb_api.feedback import FeedbackService,create_feedback_app
from ptb_api.web_admin import WebOperations,create_web_admin_router
from ptb_api.web_ops_models import Decision
from ptb_api.feedback_web_ops import WebOpsGateway
from test_web_workspaces_postgres import web,device,ORIGIN
from test_p07_policy_postgres import legacy
from web_postgres_support import cluster


def test_admin_approval_monitoring_and_revoked_device(web,tmp_path):
    e=web;user,_,_,_=device(e)
    application=user.post('/api/v1/workspaces/capacity-request',json=dict(requested_bytes=2_000_000_000,reason='合成材料',duration='30d')).json()
    feedback=FeedbackService(tmp_path/'feedback.sqlite',secret='s'*40,admin_token='m'*40)
    feedback.web_operations=WebOperations(e.workspaces,e.jobs)
    # Feedback router catches only its own safe errors in standalone production.
    app=create_feedback_app(feedback,background=False)
    from ptb_api.quota import StorageError
    from starlette.responses import JSONResponse
    @app.exception_handler(StorageError)
    async def storage_error(request,error):return JSONResponse({'detail':error.code},status_code=error.status)
    with TestClient(app) as admin:
        for path in ('status','requests'):
            assert admin.get('/api/v1/feedback/admin/web/'+path).status_code==401
        admin.headers['Authorization']='Bearer '+'m'*40
        response=admin.get('/api/v1/feedback/admin/web/status');assert response.status_code==200,response.text
        snapshot=response.json();assert snapshot['pending_requests']==1 and snapshot['online_accounts']==1
        assert snapshot['disk_free_bytes']>0 and snapshot['default_quota_bytes']==1_000_000_000
        assert response.headers['cache-control']=='no-store'
        rows=admin.get('/api/v1/feedback/admin/web/requests').json()['requests']
        assert rows[0]['reason']=='合成材料' and rows[0]['username']=='synthetic'
        body=dict(request_id=application['id'],granted_bytes=1_500_000_000,duration='7d',note='批准7天，请及时下载。')
        r=admin.post('/api/v1/feedback/admin/web/decision',json=body);assert r.status_code==200,r.text
        assert r.json()['granted_duration']=='7d'
        assert admin.post('/api/v1/feedback/admin/web/decision',json=body).status_code==409
        assert user.get('/api/v1/workspaces/capacity-request').json()['decision_note']==body['note']
        assert user.get('/api/v1/storage/usage').json()['quota_bytes']==1_500_000_000
        assert admin.post('/api/v1/feedback/admin/web/quota',json={'quota_bytes':900_000_000,'owner_id':e.owner}).status_code==200
        assert user.get('/api/v1/storage/usage').json()['quota_bytes']==900_000_000
        pair=feedback.admin_devices.start_pairing();phone=feedback.admin_devices.complete_pairing(pair['code'],'test')
        admin.headers['Authorization']='Bearer '+phone['token']
        assert admin.get('/api/v1/feedback/admin/web/status').status_code==200
        feedback.admin_devices.revoke(phone['deviceId'])
        assert admin.get('/api/v1/feedback/admin/web/status').status_code==401
        assert admin.post('/api/v1/feedback/admin/web/quota',json={'quota_bytes':2_000_000_000}).status_code==401


def test_internal_admin_key_is_separate_and_not_in_public_contract(web):
    e=web
    e.app.include_router(create_web_admin_router(WebOperations(e.workspaces,e.jobs),'i'*40))
    user,_,_,_=device(e)
    assert user.get('/api/v1/_ops/status').status_code==401
    assert user.get('/api/v1/_ops/status',headers={'Authorization':'Bearer '+'m'*40}).status_code==401
    r=user.get('/api/v1/_ops/status',headers={'Authorization':'Bearer '+'i'*40});assert r.status_code==200,r.text
    assert '/api/v1/_ops/status' not in e.app.openapi()['paths']
    assert user.post('/api/v1/_ops/quota',content=b'x'*17000,headers={'Authorization':'Bearer '+'i'*40}).status_code==413


def test_feedback_without_web_backend_remains_usable(tmp_path):
    s=FeedbackService(tmp_path/'feedback.sqlite',secret='s'*40,admin_token='m'*40)
    with TestClient(create_feedback_app(s,background=False)) as c:
        c.headers['Authorization']='Bearer '+'m'*40
        assert c.get('/api/v1/feedback/admin/web/status').status_code==503
        assert c.get('/api/v1/feedback/admin/conversations').status_code==200


def test_real_loopback_bridge_to_postgres(web,tmp_path):
    import socket,threading,time
    import uvicorn
    from ptb_api.feedback import FeedbackError
    e=web;user,_,_,_=device(e)
    request=user.post('/api/v1/workspaces/capacity-request',json=dict(requested_bytes=2_000_000_000,reason='loopback fixture',duration='30d')).json()
    e.app.include_router(create_web_admin_router(WebOperations(e.workspaces,e.jobs),'k'*40))
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(e.app,log_level='critical',access_log=False,timeout_graceful_shutdown=1))
    thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);thread.start()
    try:
        deadline=time.monotonic()+10
        while not server.started and time.monotonic()<deadline:time.sleep(.02)
        assert server.started
        config={'url':f'http://127.0.0.1:{port}/api/v1/_ops','origin':ORIGIN,'key':'k'*40}
        gateway=WebOpsGateway(config)
        assert gateway.snapshot()['pending_requests']==1
        assert gateway.requests()['requests'][0]['id']==request['id']
        decision=Decision(request_id=request['id'],granted_bytes=1_500_000_000,duration='7d',note='loopback approved')
        assert gateway.decision(decision)['decision_note']=='loopback approved'
        assert user.get('/api/v1/storage/usage').json()['quota_bytes']==1_500_000_000
        with pytest.raises(FeedbackError) as repeated:gateway.decision(decision)
        assert repeated.value.code=='capacity_request_closed'
        with pytest.raises(FeedbackError) as denied:WebOpsGateway(dict(config,key='x'*40)).snapshot()
        assert denied.value.code=='web_operations_unavailable'
    finally:
        server.should_exit=True;thread.join(10);sock.close()
    assert not thread.is_alive()


@pytest.mark.parametrize('url',['https://external.test/api/v1/_ops','http://127.0.0.1:1/other','http://user@127.0.0.1:1/api/v1/_ops'])
def test_bridge_rejects_unreviewed_destinations(url):
    with pytest.raises(ValueError):WebOpsGateway({'url':url,'origin':'https://www.phonetictoolbox.com','key':'k'*40})
