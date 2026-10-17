import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone,timedelta
from io import BytesIO
import re
import secrets
import sqlite3
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from ptb_api.feedback import FeedbackService,FeedbackError,create_feedback_app

ADMIN='administrator-test-only-key-123456789'
SECRET='private-hmac-test-only-key-123456789'


class Mailbox:
    def __init__(self):
        self.sent=[]
        self.fail=False
    def send(self,*args):
        if self.fail: raise OSError('private detail must not be returned')
        self.sent.append(args)


@pytest.fixture
def service(tmp_path):
    now=[datetime(2026,10,10,10,tzinfo=timezone(timedelta(hours=8))).timestamp()]
    mail=Mailbox()
    s=FeedbackService(tmp_path/'feedback.sqlite',secret=SECRET,admin_token=ADMIN,mailer=mail,clock=lambda:now[0])
    s.now,s.mail=now,mail
    return s


def identity(s,email='first@example.org'):
    v=s.start_verification(email)
    assert s.deliver_one()
    code=re.search(r'验证码：(\d{6})',s.mail.sent[-1][2]).group(1)
    return s.verify(v['challenge'],code)


def picture():
    b=BytesIO()
    Image.new('RGB',(24,12),(44,88,122)).save(b,format='PNG')
    return {'data':base64.b64encode(b.getvalue()).decode()}


def test_trash_restore_keeps_messages_images_provenance_and_original_deadline(service):
    user=identity(service)
    other=identity(service,'other@example.org')
    first=service.send(user['owner'],'user','保留反馈',[picture()],'trash-user-1',{'source':'web'})
    service.send(other['owner'],'user','其他会话',[],'trash-other-1')
    service.set_status(user['owner'],'resolved')
    with service.db() as db:
        original_schema=[tuple(r) for r in db.execute('SELECT sql FROM sqlite_master ORDER BY name')]
    trash=service.trash_conversation(user['owner'],first['id'])
    assert trash['purgeAt']-trash['deletedAt']==30*86400
    assert [r['id'] for r in service.conversations()]==[other['owner']]
    archived=next(r for r in service.conversations(True) if r['id']==user['owner'])
    assert archived['trashed'] and archived['client']=={'source':'web'}
    service.now[0]+=86400
    assert service.trash_conversation(user['owner'],first['id'])==trash  # Retries do not extend 30 days.
    before=service.messages(user['owner'])
    assert before['trashed'] and before['messages'][0]['images']
    restarted=FeedbackService(service.database,secret=SECRET,admin_token=ADMIN,mailer=service.mail,clock=service.clock)
    assert restarted.messages(user['owner'])['purgeAt']==trash['purgeAt']
    assert restarted.restore_conversation(user['owner'])=={'trashed':False}
    assert restarted.restore_conversation(user['owner'])=={'trashed':False}
    after=restarted.messages(user['owner'])
    assert not after['trashed'] and after['messages']==before['messages'] and after['status']=='resolved'
    assert restarted.authenticate(user['token'])==user['owner']
    with restarted.db() as db:
        assert [tuple(r) for r in db.execute('SELECT sql FROM sqlite_master ORDER BY name')]==original_schema


def test_trash_rejects_stale_confirmation_and_blocks_admin_writes(service):
    user=identity(service)
    first=service.send(user['owner'],'user','第一条',[],'trash-stale-1')
    last=service.send(user['owner'],'user','刚刚补充',[],'trash-stale-2')
    with pytest.raises(FeedbackError,match='conversation_changed'):
        service.trash_conversation(user['owner'],first['id'])
    service.trash_conversation(user['owner'],last['id'])
    with pytest.raises(FeedbackError,match='conversation_trashed'):
        service.send(user['owner'],'admin','不可回复',[],'trash-admin-1')
    with pytest.raises(FeedbackError,match='conversation_trashed'):
        service.set_status(user['owner'],'open')
    assert service.daily_check() is None  # Recycled unread messages do not produce daily reminders.
    service.send(user['owner'],'user','刚刚补充',[],'trash-stale-2')
    assert service.messages(user['owner'])['trashed']  # A retry is not a new user message.
    service.send(user['owner'],'user','新的反馈',[],'trash-stale-3')
    assert not service.messages(user['owner'])['trashed']
    assert len(service.messages(user['owner'])['messages'])==3


def test_trash_cancels_pending_mail_and_refuses_inflight_delivery(service):
    user=identity(service)
    mid=service.send(user['owner'],'admin','邮件回复',[picture()],'trash-mail-1')['id']
    with service.db() as db:
        db.execute("UPDATE outbox SET state='sending',lease_until=? WHERE message=?",(service.clock()+60,mid))
    with pytest.raises(FeedbackError,match='conversation_mail_busy'):
        service.trash_conversation(user['owner'],mid)
    with service.db() as db:
        db.execute("UPDATE outbox SET state='pending',lease_until=0 WHERE message=?",(mid,))
    service.trash_conversation(user['owner'],mid)
    with pytest.raises(FeedbackError,match='conversation_trashed'):
        service.retry_mail(mid)
    assert not service.deliver_one()
    message=service.messages(user['owner'])['messages'][0]
    assert message['mailState']=='skipped' and message['mailError']=='conversation_trashed'
    assert message['images']
    service.restore_conversation(user['owner'])
    assert not service.deliver_one()  # Restoring does not unexpectedly send old queued emails.


def test_trash_expires_at_30_days_and_clears_only_target_history(service):
    user=identity(service)
    other=identity(service,'other@example.org')
    service.send(user['owner'],'user','用户图片',[picture()],'trash-expiry-1',{'source':'desktop','version':'3.0.2','packageKind':'portable'})
    last=service.send(user['owner'],'admin','管理员图片',[picture()],'trash-expiry-2')['id']
    kept=service.send(other['owner'],'user','其他图片',[picture()],'trash-expiry-other')['id']
    trash=service.trash_conversation(user['owner'],last)
    service.now[0]=trash['purgeAt']-1
    assert service.purge_trash()=={'purged':0}
    assert service.messages(user['owner'])['messages']
    service.now[0]=trash['purgeAt']
    with pytest.raises(FeedbackError,match='conversation_unavailable'):
        service.restore_conversation(user['owner'])
    assert service.messages(user['owner'])['messages']==[]
    with pytest.raises(FeedbackError,match='conversation_unavailable'):
        service.send(user['owner'],'admin','过期旧窗口回复',[],'trash-expiry-stale-reply')
    assert service.authenticate(user['token'])==user['owner']
    with service.db() as db:
        assert db.execute('SELECT COUNT(*) FROM messages WHERE owner=?',(user['owner'],)).fetchone()[0]==0
        assert db.execute('SELECT COUNT(*) FROM attachments').fetchone()[0]==1
        assert db.execute('SELECT COUNT(*) FROM outbox WHERE message IS NOT NULL').fetchone()[0]==0
        owner_events=[__import__('json').loads(r[0]) for r in db.execute("SELECT payload FROM events WHERE json_extract(payload,'$.owner')=?",(user['owner'],))]
        assert len(owner_events)==1 and owner_events[0]['kind']=='conversation_deleted'
    assert service.messages(other['owner'])['messages'][0]['id']==kept
    service.send(user['owner'],'user','重新留言',[],'trash-expiry-fresh')
    assert [m['text'] for m in service.messages(user['owner'])['messages']]==['重新留言']
    assert service.purge_trash()=={'purged':0}


def test_background_maintenance_purges_expired_trash_on_restart(service,monkeypatch):
    import asyncio
    from ptb_api.feedback import maintain_feedback
    user=identity(service)
    mid=service.send(user['owner'],'user','后台到期',[],'trash-background-1')['id']
    service.trash_conversation(user['owner'],mid)
    service.now[0]+=30*86400
    restarted=FeedbackService(service.database,secret=SECRET,admin_token=ADMIN,clock=service.clock)
    async def stop_after_first_cycle(seconds):
        raise asyncio.CancelledError()
    monkeypatch.setattr('ptb_api.feedback.asyncio.sleep',stop_after_first_cycle)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(maintain_feedback(restarted))
    with restarted.db() as db:
        assert db.execute('SELECT COUNT(*) FROM messages').fetchone()[0]==0


def test_trash_http_requires_admin_and_supports_device_credentials(service):
    user=identity(service)
    mid=service.send(user['owner'],'user','权限验证',[],'trash-auth-1')['id']
    device=service.admin_devices.complete_pairing(service.admin_devices.start_pairing()['code'],'trash-test')
    path='/api/v1/feedback/admin/conversations/'+user['owner']
    with TestClient(create_feedback_app(service,background=False)) as api:
        health=api.get('/api/v1/feedback/health').json()
        assert health['conversationTrash'] is True and health['trashRetentionDays']==30
        for headers in [{},{'Authorization':'Bearer '+user['token']}]:
            assert api.post(path+'/trash',headers=headers,json={'through':mid}).status_code==401
            assert api.post(path+'/restore',headers=headers,json={}).status_code==401
        headers={'Authorization':'Bearer '+device['token']}
        assert api.post(path+'/trash',headers=headers,json={'through':0}).status_code==422
        assert api.post(path+'/trash',headers=headers,json={'through':mid}).status_code==200
        assert api.get('/api/v1/feedback/admin/conversations',headers=headers).json()['conversations']==[]
        assert api.get('/api/v1/feedback/admin/conversations?includeTrash=true',headers=headers).json()['conversations'][0]['trashed']
        assert api.post(path+'/restore',headers=headers,json={}).status_code==200
        service.admin_devices.revoke(device['deviceId'])
        assert api.post(path+'/trash',headers=headers,json={'through':mid}).status_code==401


def test_provenance_is_per_message_durable_and_retry_preserves_original(service):
    user=identity(service)
    owner=user['owner']
    with service.db() as db:
        schema=db.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
        schema=[tuple(row) for row in schema]
    first=service.send(owner,'user','升级前',[],'provenance-first',{'source':'desktop','version':'3.0.1','packageKind':'installer'})
    retry=service.send(owner,'user','升级前',[],'provenance-first',{'source':'desktop','version':'3.0.2','packageKind':'portable'})
    assert retry['duplicate'] and retry['id']==first['id']
    service.send(owner,'user','升级后',[],'provenance-second',{'source':'desktop','version':'3.0.2','packageKind':'portable'})
    service.send(owner,'user','从网页补充',[],'provenance-third',{'source':'web'})
    service.send(owner,'admin','收到',[],'provenance-reply')
    restarted=FeedbackService(service.database,secret=SECRET,admin_token=ADMIN,mailer=service.mail,clock=service.clock)
    messages=restarted.messages(owner)['messages']
    assert [m['client'] for m in messages]==[
        {'source':'desktop','version':'3.0.1','packageKind':'installer'},
        {'source':'desktop','version':'3.0.2','packageKind':'portable'}, {'source':'web'},None]
    assert restarted.conversations()[0]['client']=={'source':'web'}
    assert restarted.messages(owner,first['id'])['messages'][0]['client']['version']=='3.0.2'
    with restarted.db() as db:
        assert [tuple(row) for row in db.execute("SELECT sql FROM sqlite_master ORDER BY name")]==schema
    restarted.send(owner,'user','旧客户端',[],'provenance-legacy')
    assert restarted.conversations()[0]['client'] is None


def test_provenance_headers_validate_and_legacy_clients_remain_compatible(service):
    user=identity(service)
    headers={'Authorization':'Bearer '+user['token']}
    with TestClient(create_feedback_app(service,background=False)) as api:
        def send(key,extra):
            return api.post('/api/v1/feedback/messages',headers={**headers,**extra},json={'text':'test','requestId':key})
        assert send('provenance-native',{'X-PTB-Feedback-Source':'desktop','X-PTB-Feedback-Version':'3.0.2','X-PTB-Feedback-Package':'installer'}).status_code==200
        assert send('provenance-web-1',{'X-PTB-Feedback-Source':'web'}).status_code==200
        assert send('provenance-old-web',{'Origin':'https://www.phonetictoolbox.com'}).status_code==200
        assert send('provenance-old-app',{'User-Agent':'PhoneticToolbox-Feedback/1'}).status_code==200
        assert send('provenance-unknown',{}).status_code==200
        for extra in [{'X-PTB-Feedback-Source':'invalid'}, {'X-PTB-Feedback-Source':'desktop','X-PTB-Feedback-Version':'<script>'}, {'X-PTB-Feedback-Source':'desktop','X-PTB-Feedback-Package':'invalid'}]:
            assert send('provenance-invalid',extra).status_code==422
        rows=api.get('/api/v1/feedback/admin/conversations/'+user['owner'],headers={'Authorization':'Bearer '+ADMIN}).json()['messages']
        assert [r['client'] for r in rows]==[{'source':'desktop','version':'3.0.2','packageKind':'installer'},{'source':'web'},{'source':'web'},{'source':'desktop'},None]


def test_mail_opt_out_cancels_queue_and_survives_restart_and_recovery(service):
    a,b=identity(service),identity(service,'other@example.org')
    pending=service.send(a['owner'],'admin','尚未投递',[],'preference-pending-1')
    service.set_mail_notifications(a['owner'],False)
    assert service.messages(a['owner'])['messages'][0]['mailState']=='skipped'
    with pytest.raises(FeedbackError,match='mail_notifications_disabled'):
        service.retry_mail(pending['id'])
    restarted=FeedbackService(service.database,secret=SECRET,admin_token=ADMIN,mailer=service.mail,clock=service.clock)
    assert not restarted.messages(a['owner'])['mailNotifications']
    assert restarted.messages(b['owner'])['mailNotifications']
    restarted.send(a['owner'],'admin','应用内继续聊天',[picture()],'preference-disabled-1')
    assert restarted.messages(a['owner'])['messages'][-1]['mailState']=='disabled'
    assert not restarted.deliver_one()
    assert len(service.mail.sent)==2  # Only the two verification emails.
    service.now[0]+=61
    # Email recovery reuses the same owner and therefore the same saved setting.
    restarted.mail=service.mail
    recovered=identity(restarted,'first@example.org')
    assert recovered['owner']==a['owner'] and not restarted.messages(a['owner'])['mailNotifications']
    restarted.set_mail_notifications(a['owner'],True)
    restarted.send(a['owner'],'admin','只提醒新的回复',[],'preference-enabled-1')
    assert restarted.deliver_one() and not restarted.deliver_one()
    assert service.mail.sent[-1][2].startswith('您的 PhoneticToolbox 反馈收到回复：\n\n只提醒新的回复')
    with restarted.db() as db:
        assert db.execute('SELECT version FROM feedback_meta').fetchone()[0]==2


def test_mail_preference_api_is_user_scoped_strict_and_authenticated(service):
    a,b=identity(service),identity(service,'other@example.org')
    client=TestClient(create_feedback_app(service,background=False))
    path='/api/v1/feedback/preferences'
    headers={'Authorization':'Bearer '+a['token']}
    assert client.post(path,json={'mailNotifications':False}).status_code==401
    assert client.post(path,json={'mailNotifications':False},headers={'Authorization':'Bearer '+ADMIN}).status_code==401
    for value in ('false',0,None):
        assert client.post(path,json={'mailNotifications':value},headers=headers).status_code==422
    assert client.post(path,json={'mailNotifications':False,'owner':b['owner']},headers=headers).status_code==422
    assert client.post(path,json={'mailNotifications':False},headers=headers).json()=={'mailNotifications':False}
    assert not service.messages(a['owner'])['mailNotifications'] and service.messages(b['owner'])['mailNotifications']


def test_mail_opt_out_skips_expired_send_lease_and_failed_retry(service):
    a=identity(service)
    service.send(a['owner'],'admin','旧的投递租约',[],'preference-lease-1')
    with service.db() as db:
        db.execute("UPDATE outbox SET state='sending',lease_until=? WHERE message IS NOT NULL",(service.now[0]-1,))
    service.set_mail_notifications(a['owner'],False)
    assert not service.deliver_one()
    assert service.messages(a['owner'])['messages'][0]['mailState']=='skipped'
    assert len(service.mail.sent)==1


def test_verified_identity_recovers_and_tokens_are_not_stored(service):
    first=identity(service)
    owner=service.authenticate(first['token'])
    service.send(owner,'user','第一条反馈',[picture()],'first-message-1')
    service.now[0]+=61
    recovered=identity(service)
    assert recovered['owner']==owner
    assert service.messages(service.authenticate(recovered['token']))['messages'][0]['text']=='第一条反馈'
    with service.db() as db:
        assert all(first['token'] not in row['hash'] for row in db.execute('SELECT hash FROM sessions'))
        tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert all(c[1] not in {'ip','client_ip','remote_ip','ip_address'} for t in tables for c in db.execute(f'PRAGMA table_info({t})'))


def test_code_attempts_and_expiry(service):
    v=service.start_verification('sample@example.org')
    for _ in range(5):
        with pytest.raises(FeedbackError,match='verification_invalid'):
            service.verify(v['challenge'],'not-the-code')
    with pytest.raises(FeedbackError,match='verification_expired'):
        service.verify(v['challenge'],'000000')
    service.now[0]+=61
    v=service.start_verification('sample@example.org')
    service.now[0]+=601
    with pytest.raises(FeedbackError,match='verification_expired'):
        service.verify(v['challenge'],'000000')


def test_api_ownership_and_admin_boundary(service):
    a,b=identity(service),identity(service,'second@example.org')
    client=TestClient(create_feedback_app(service,background=False))
    ah={'Authorization':'Bearer '+a['token']}
    bh={'Authorization':'Bearer '+b['token']}
    body={'text':'仅第一用户可见','images':[picture()],'requestId':'private-message-1'}
    assert client.post('/api/v1/feedback/messages',json=body,headers=ah).status_code==200
    assert client.get('/api/v1/feedback/conversation',headers=bh).json()['messages']==[]
    assert client.get('/api/v1/feedback/admin/conversations/'+a['owner'],headers=bh).status_code==401
    assert client.get('/api/v1/feedback/conversation').status_code==401
    assert client.get('/api/v1/feedback/conversation',headers=ah).headers['cache-control']=='no-store'
    assert client.get('/api/v1/feedback/admin/conversations',headers={'Authorization':'Bearer '+ADMIN}).json()['conversations'][0]['unread']==1
    assert client.post('/api/v1/feedback/verification/complete',json={'challenge':'invalid','code':'private'}).json()=={'detail':'invalid_request'}


def test_idempotent_reply_and_atomic_mail_queue(service):
    a=identity(service)
    def send(): return service.send(a['owner'],'admin','已修复，感谢反馈。',[picture()],'reply-message-1')
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:send(),range(4)))
    assert len({r['id'] for r in results})==1
    conversation=service.messages(a['owner'])
    assert len(conversation['messages'])==1
    assert conversation['messages'][0]['mailState']=='pending'
    with service.db() as db:
        assert db.execute('SELECT COUNT(*) FROM outbox WHERE message IS NOT NULL').fetchone()[0]==1
    with pytest.raises(FeedbackError,match='message_key_conflict'):
        service.send(a['owner'],'admin','different',[],'reply-message-1')


def test_smtp_failure_keeps_app_reply_and_retries(service):
    a=identity(service)
    service.send(a['owner'],'admin','应用里仍然可见',[],'mail-reply-1')
    service.mail.fail=True
    assert service.deliver_one()
    m=service.messages(a['owner'])['messages'][0]
    assert m['text']=='应用里仍然可见' and m['mailState']=='pending'
    assert m['mailError']=='smtp_delivery_failed'
    service.now[0]+=31
    service.mail.fail=False
    assert service.deliver_one()
    assert service.messages(a['owner'])['messages'][0]['mailState']=='sent'
    with service.db() as db:
        assert db.execute('SELECT body FROM outbox WHERE message IS NOT NULL').fetchone()[0]==''


def test_crashed_outbox_lease_is_recovered(service):
    a=identity(service)
    service.send(a['owner'],'admin','重启后发信',[],'crash-reply-1')
    with service.db() as db:
        db.execute("UPDATE outbox SET state='sending',lease_until=? WHERE message IS NOT NULL",(service.now[0]-1,))
    assert service.deliver_one()
    assert service.messages(a['owner'])['messages'][0]['mailState']=='sent'


def test_server_daily_check_once_and_offline_backlog_survives_restart(service):
    a=identity(service)
    service.send(a['owner'],'user','问题一',[],'daily-message-1')
    event=service.daily_check()
    assert event['count']==1 and event['conversations']==1
    assert service.daily_check() is None
    new=FeedbackService(service.database,secret=SECRET,admin_token=ADMIN,mailer=service.mail,clock=service.clock)
    assert [e for e in new.events(0) if e['kind']=='daily_feedback']==[event]
    assert new.daily_check() is None
    service.send(a['owner'],'user','追加留言',[],'daily-message-2')
    assert new.daily_check() is None
    service.now[0]+=86400
    assert new.daily_check()['count']==1
    assert len([e for e in new.events(0) if e['kind']=='daily_feedback'])==2
    assert len([e for e in new.events(event['id']) if e['kind']=='daily_feedback'])==1


def test_handled_feedback_does_not_notify_and_schedule_is_shanghai(service):
    a=identity(service)
    sent=service.send(a['owner'],'user','已处理',[],'seen-message-1')
    service.mark_read(a['owner'],'user',sent['id'])
    assert service.daily_check() is None
    service.now[0]=datetime(2026,10,11,0,tzinfo=timezone.utc).timestamp()
    service.send(a['owner'],'user','次日早八点',[],'seen-message-2')
    assert service.daily_check() is None
    service.now[0]+=3600
    assert service.daily_check()['day']=='2026-10-11'


def test_invalid_images_and_oversized_requests_do_not_publish(service):
    a=identity(service)
    with pytest.raises(FeedbackError,match='invalid_image'):
        service.send(a['owner'],'user','bad',[{'data':base64.b64encode(b'<svg onload="bad"/>').decode()}],'bad-message-1')
    assert service.messages(a['owner'])['messages']==[]
    client=TestClient(create_feedback_app(service,background=False))
    response=client.post('/api/v1/feedback/messages',content=b'x'*14_000_001,headers={'Authorization':'Bearer '+a['token'],'Content-Type':'application/json'})
    assert response.status_code in (413,400)
    assert 'x'*50 not in response.text


def test_missing_mail_is_explicit_and_rate_limits_survive_restart(tmp_path,service):
    no=FeedbackService(tmp_path/'none.sqlite',secret=SECRET,admin_token=ADMIN)
    with pytest.raises(FeedbackError,match='mail_not_configured'):
        no.start_verification('first@example.org')
    service.start_verification('first@example.org')
    new=FeedbackService(service.database,secret=SECRET,admin_token=ADMIN,mailer=service.mail,clock=service.clock)
    with pytest.raises(FeedbackError,match='verification_rate_limited'):
        new.start_verification('first@example.org')


def test_existing_unknown_database_is_not_modified(tmp_path):
    target=tmp_path/'existing-research.sqlite'
    with sqlite3.connect(target) as db:
        db.execute('CREATE TABLE research(value TEXT)')
        db.execute("INSERT INTO research VALUES('keep original')")
    original=target.read_bytes()
    with pytest.raises(ValueError,match='not a feedback'):
        FeedbackService(target,secret=SECRET,admin_token=ADMIN)
    assert target.read_bytes()==original


def test_contact_email_never_recovers_an_unverified_conversation(tmp_path,service):
    service.allow_unverified=True
    token=secrets.token_urlsafe(32)
    first=service.open_device('same@example.org',token)
    assert service.open_device('same@example.org',token)['owner']==first['owner']
    service.send(first['owner'],'user','这条消息只属于第一台设备',[],'device-message-1')
    second=service.open_device('same@example.org',secrets.token_urlsafe(32))
    assert second['owner']!=first['owner']
    assert service.messages(second['owner'])['messages']==[]
    verified=identity(service,'same@example.org')
    assert verified['owner'] not in (first['owner'],second['owner'])
    assert service.messages(verified['owner'])['messages']==[]
    service.send(first['owner'],'admin','邮箱交由维护者手动发送',[],'device-reply-1')
    assert service.messages(first['owner'])['messages'][-1]['mailState']=='manual'
    with service.db() as db:
        assert db.execute('SELECT COUNT(*) FROM outbox WHERE message IS NOT NULL').fetchone()[0]==0
        assert token.encode() not in service.database.read_bytes()


def test_device_flow_without_smtp_and_later_verified_promotion(tmp_path,service):
    no=FeedbackService(tmp_path/'device.sqlite',secret=SECRET,admin_token=ADMIN,allow_unverified=True)
    client=TestClient(create_feedback_app(no,background=False))
    assert client.get('/api/v1/feedback/health').json()['allowUnverified']
    opened=client.post('/api/v1/feedback/identity',json={'email':'first@example.org','deviceToken':secrets.token_urlsafe(32)}).json()
    owner=no.authenticate(opened['token'])
    no.send(owner,'admin','只发送到应用',[],'no-mail-reply-1')
    assert no.messages(owner)['messages'][0]['mailState']=='manual'
    assert not no.deliver_one()
    with no.db() as db:
        assert db.execute('SELECT COUNT(*) FROM outbox').fetchone()[0]==0
    no.mailer=service.mail
    challenge=no.start_verification('first@example.org')
    no.deliver_one()
    code=re.search(r'验证码：(\d{6})',service.mail.sent[-1][2]).group(1)
    promoted=no.verify(challenge['challenge'],code,owner)
    assert promoted['owner']==owner and promoted['emailVerified']
    assert no.messages(owner)['messages'][0]['text']=='只发送到应用'
    assert no.messages(owner)['messages'][0]['mailState']=='manual'


def test_device_mode_closed_by_default_and_limits_persist(tmp_path):
    service=FeedbackService(tmp_path/'closed.sqlite',secret=SECRET,admin_token=ADMIN)
    with pytest.raises(FeedbackError,match='email_verification_required'):
        service.open_device('first@example.org',secrets.token_urlsafe(32))
    service.allow_unverified=True
    for index in range(20):
        service.open_device(f'user{index}@example.org',secrets.token_urlsafe(32))
    restarted=FeedbackService(service.database,secret=SECRET,admin_token=ADMIN,allow_unverified=True)
    with pytest.raises(FeedbackError,match='identity_rate_limited'):
        restarted.open_device('again@example.org',secrets.token_urlsafe(32))


def test_verification_never_silently_replaces_current_device_history(service):
    service.allow_unverified=True
    device=service.open_device('first@example.org',secrets.token_urlsafe(32))
    service.send(device['owner'],'user','保留当前设备会话',[],'keep-device-history')
    official=identity(service)
    assert official['owner']!=device['owner']
    service.now[0]+=61
    challenge=service.start_verification('first@example.org')
    service.deliver_one()
    code=re.search(r'验证码：(\d{6})',service.mail.sent[-1][2]).group(1)
    with pytest.raises(FeedbackError,match='email_identity_conflict'):
        service.verify(challenge['challenge'],code,device['owner'])
    assert service.authenticate(device['token'])==device['owner']
    assert service.messages(device['owner'])['messages'][0]['text']=='保留当前设备会话'


def test_inactive_device_identity_keeps_access_without_email_recovery(service):
    service.allow_unverified=True
    token=secrets.token_urlsafe(32)
    device=service.open_device('contact@example.org',token)
    verified=identity(service,'verified@example.org')
    service.now[0]+=181*86400
    assert service.open_device('contact@example.org',token)['owner']==device['owner']
    assert service.authenticate(token)==device['owner']
    with pytest.raises(FeedbackError,match='feedback_unauthorized'):
        service.authenticate(verified['token'])
