import json
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi.testclient import TestClient
from ptb_api.feedback import FeedbackService, FeedbackError, create_feedback_app

ADMIN='device-test-master-12345678901234567890'
SECRET='device-test-hmac-12345678901234567890'

@pytest.fixture
def service(tmp_path):
    now=[1900000000.0]
    service=FeedbackService(tmp_path/'feedback.sqlite',secret=SECRET,admin_token=ADMIN,allow_unverified=True,clock=lambda:now[0])
    service.now=now
    return service

def test_pairing_one_use_revoke_and_restart(service):
    code=service.admin_devices.start_pairing()['code']
    device=service.admin_devices.complete_pairing(code,'红米')
    assert service.authenticate(device['token'],True) is None
    with pytest.raises(FeedbackError):service.admin_devices.complete_pairing(code,'duplicate')
    raw=service.admin_devices.path.read_text(encoding='utf-8')
    assert code not in raw and device['token'] not in raw
    restarted=FeedbackService(service.database,secret=SECRET,admin_token=ADMIN,clock=service.clock)
    assert restarted.admin_devices.list_devices()['devices'][0]['name']=='红米'
    restarted.admin_devices.revoke(device['deviceId'])
    with pytest.raises(FeedbackError,match='admin_unauthorized'):service.authenticate(device['token'],True)
    assert service.authenticate(ADMIN,True) is None

def test_pair_attempt_limit_expiry_and_concurrent_consumption(service):
    code=service.admin_devices.start_pairing()['code']
    wrong='00000000' if code!='00000000' else '11111111'
    for _ in range(5):
        with pytest.raises(FeedbackError):service.admin_devices.complete_pairing(wrong,'wrong')
    with pytest.raises(FeedbackError):service.admin_devices.complete_pairing(code,'locked')
    code=service.admin_devices.start_pairing()['code']
    service.now[0]+=301
    with pytest.raises(FeedbackError):service.admin_devices.complete_pairing(code,'expired')
    code=service.admin_devices.start_pairing()['code']
    def consume(_):
        try:return service.admin_devices.complete_pairing(code,'concurrent')
        except FeedbackError:return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(value is not None for value in pool.map(consume,range(4)))==1

def test_device_api_and_invalid_requests(service):
    client=TestClient(create_feedback_app(service,background=False))
    prefix='/api/v1/feedback/admin/devices'
    admin={'Authorization':'Bearer '+ADMIN}
    assert client.post(prefix+'/pairing').status_code==401
    code=client.post(prefix+'/pairing',headers=admin).json()['code']
    assert client.post(prefix+'/pairing/complete',json={'code':code,'name':'phone','token':'bad'}).status_code==422
    device=client.post(prefix+'/pairing/complete',json={'code':code,'name':'phone'}).json()
    paired={'Authorization':'Bearer '+device['token']}
    assert client.get('/api/v1/feedback/admin/conversations',headers=paired).status_code==200
    assert 'token' not in client.get(prefix,headers=paired).text
    assert client.post(prefix+'/'+device['deviceId']+'/revoke',headers=admin).status_code==200
    assert client.get(prefix,headers=paired).status_code==401

def test_instant_events_atomic_idempotent_and_offline(service):
    import secrets
    identity=service.open_device('fixture@example.org',secrets.token_urlsafe(32))
    def send(_):return service.send(identity['owner'],'user','新留言',[],'instant-message-1')
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(send,range(4)))
    events=service.events(0)
    assert len(events)==1 and events[0]['kind']=='new_feedback'
    assert events[0]['owner']==identity['owner'] and events[0]['messageId']==results[0]['id']
    service.send(identity['owner'],'admin','回复',[],'instant-reply-1')
    restarted=FeedbackService(service.database,secret=SECRET,admin_token=ADMIN,clock=service.clock)
    assert restarted.events(events[0]['id'])[0]['kind']=='conversation_changed'
    with pytest.raises(FeedbackError):service.send(identity['owner'],'user','invalid',[{'data':'bad'}],'invalid-image-1')
    assert len(restarted.events(0))==2
