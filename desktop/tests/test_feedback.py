from pathlib import Path
import pytest
from ptb_desktop.feedback import FeedbackClient,FeedbackClientError,protected,valid_base
from ptb_api.feedback_mail import settings_from_script


@pytest.mark.parametrize('status',[404,405])
@pytest.mark.parametrize('operation',['trash','restore'])
def test_legacy_server_missing_trash_route_has_an_actionable_error(tmp_path,status,operation):
    from io import BytesIO
    from urllib.error import HTTPError
    client=FeedbackClient(tmp_path,admin=True)
    def missing(request,timeout):
        raise HTTPError(request.full_url,status,'missing',None,BytesIO(b'{"detail":"Not Found"}'))
    client.opener.open=missing
    with pytest.raises(FeedbackClientError,match='回收站尚未在此服务器启用'):
        client.call('/admin/conversations/test-owner/'+operation,{'through':1} if operation=='trash' else {})


def test_actual_missing_conversation_is_not_reported_as_a_server_upgrade(tmp_path):
    from io import BytesIO
    from urllib.error import HTTPError
    client=FeedbackClient(tmp_path,admin=True)
    def missing(request,timeout):
        raise HTTPError(request.full_url,404,'missing',None,BytesIO(b'{"detail":"conversation_unavailable"}'))
    client.opener.open=missing
    with pytest.raises(FeedbackClientError,match='会话已不存在'):
        client.call('/admin/conversations/test-owner/trash',{'through':1})


def test_manager_native_actions_include_trash_and_require_admin(tmp_path):
    calls=[]
    client=FeedbackClient(tmp_path,admin=True)
    client.call=lambda path,body=None,**kw:calls.append((path,body)) or {'conversations':[]}
    client.request('conversations',{})
    client.request('trashConversation',{'owner':'owner-test-123','through':12})
    client.request('restoreConversation',{'owner':'owner-test-123'})
    assert calls==[('/admin/conversations?includeTrash=true',None),('/admin/conversations/owner-test-123/trash',{'through':12}),('/admin/conversations/owner-test-123/restore',{})]
    with pytest.raises(FeedbackClientError):client.request('trashConversation',{'owner':'owner-test-123','through':0})
    user=FeedbackClient(tmp_path/'user')
    with pytest.raises(FeedbackClientError):user.request('trashConversation',{'owner':'owner-test-123','through':12})
    with pytest.raises(FeedbackClientError):user.request('restoreConversation',{'owner':'owner-test-123'})


def test_native_title_tracks_current_unread_and_clears_when_read():
    from ptb_desktop.feedback_manager import FeedbackManager
    class Window:
        tray=None
        def setWindowTitle(self,value):self.title=value
    window=Window()
    FeedbackManager.update_inbox_title(window,3,1)
    assert window.title=='PhoneticToolbox 反馈管理 · 1 个会话有 3 条未读留言'
    FeedbackManager.update_inbox_title(window,0,0)
    assert window.title=='PhoneticToolbox 反馈管理'
    FeedbackManager.notify(window,'{"kind":"new_feedback","conversations":1,"count":3}')
    assert window.title=='PhoneticToolbox 反馈管理'  # Events do not overwrite the current unread snapshot.


def test_bridge_revalidates_historical_events_and_excludes_recycle_bin(tmp_path):
    import json
    from ptb_desktop.feedback_bridge import FeedbackBridge
    class Client:
        rows=[{'id':'active-user','unread':0},{'id':'trash-user','unread':20,'trashed':True}]
        def subscribe(self,callback):pass
        def close(self):pass
        def request(self,op,args):return {'conversations':self.rows}
    client=Client()
    bridge=FeedbackBridge(tmp_path,admin=True,client=client)
    counts=[];events=[]
    bridge.inboxState.connect(lambda count,conversations:counts.append((count,conversations)))
    bridge.notification.connect(lambda value:events.append(json.loads(value)))
    try:
        bridge.publish_event({'kind':'daily_feedback','count':3,'conversations':1})
        assert counts[-1]==(0,0) and events[-1]['count']==0
        client.rows[0]['unread']=2
        bridge.publish_event({'kind':'new_feedback','owner':'active-user','count':1,'conversations':1})
        assert counts[-1]==(2,1) and events[-1]['count']==2
        client.rows[0]['unread']=0
        bridge.publish_event({'kind':'conversation_changed','owner':'active-user'})
        assert counts[-1]==(0,0)
    finally:bridge.close()


@pytest.mark.parametrize('kind',['installer','portable','source'])
def test_user_message_headers_report_host_version_without_changing_legacy_body(tmp_path,kind):
    import json
    from io import BytesIO
    client=FeedbackClient(tmp_path,client_info={'source':'desktop','version':'3.0.2','packageKind':kind})
    captured=[]
    def open_request(request,timeout):
        captured.append(request)
        response=BytesIO(b'{"id":1,"duplicate":false}')
        response.geturl=lambda:request.full_url
        return response
    client.opener.open=open_request
    body={'text':'test','images':[],'requestId':'provenance-native-send'}
    client.request('send',body)
    request=captured[0]
    assert json.loads(request.data)==body
    headers={k.lower():v for k,v in request.header_items()}
    assert headers['x-ptb-feedback-source']=='desktop'
    assert headers['x-ptb-feedback-version']=='3.0.2'
    assert headers['x-ptb-feedback-package']==kind
    assert not client.admin


def test_manager_reply_does_not_report_user_provenance(tmp_path):
    from io import BytesIO
    client=FeedbackClient(tmp_path,admin=True)
    captured=[]
    def open_request(request,timeout):
        captured.append(request)
        response=BytesIO(b'{"id":1}')
        response.geturl=lambda:request.full_url
        return response
    client.opener.open=open_request
    client.request('send',{'owner':'test-owner-123','text':'reply','images':[],'requestId':'provenance-admin-send'})
    assert not any(k.lower().startswith('x-ptb-feedback-') for k,v in captured[0].header_items())


def test_windows_credentials_are_protected_and_restored(tmp_path):
    client=FeedbackClient(tmp_path)
    token='only-a-private-test-token-123456789'
    client.state={'token':token,'email':'sample@example.org','cursor':12}
    client.save()
    assert token.encode() not in client.path.read_bytes()
    assert b'sample@example.org' not in client.path.read_bytes()
    recovered=FeedbackClient(tmp_path)
    assert recovered.state['token']==token
    assert recovered.state['cursor']==12
    assert 'token' not in recovered.status()


@pytest.mark.parametrize('url',['http://public.test/api/v1/feedback','https://user:password@site.test/api/v1/feedback','https://site.test/api/v1/feedback?token=secret','https://site.test/arbitrary','file:///api/v1/feedback'])
def test_server_url_rejects_plaintext_and_embedded_credentials(url):
    with pytest.raises(FeedbackClientError):valid_base(url)


def test_mail_source_is_only_statically_parsed(tmp_path):
    marker=tmp_path/'must-not-exist'
    source=tmp_path/'source.py'
    source.write_text("SMTP_SERVER='smtp.test'\nSMTP_PORT=465\nSENDER_EMAIL='sample@example.org'\nSENDER_PASSWORD='test-only-private'\nfrom pathlib import Path\nPath("+repr(str(marker))+ ").write_text('executed')\n",encoding='utf-8')
    settings=settings_from_script(source)
    assert settings['host']=='smtp.test' and settings['ssl'] is True
    assert not marker.exists()


def test_corrupt_identity_does_not_get_overwritten(tmp_path):
    identity=tmp_path/'identity.bin'
    identity.write_bytes(b'corrupt private state')
    with pytest.raises(FeedbackClientError):FeedbackClient(tmp_path)
    assert identity.read_bytes()==b'corrupt private state'


def test_device_enrollment_retries_with_same_protected_key(tmp_path):
    client=FeedbackClient(tmp_path)
    attempts=[]
    def interrupted(path,body,**kwargs):
        attempts.append(body.copy())
        raise FeedbackClientError('network interrupted')
    client.call=interrupted
    with pytest.raises(FeedbackClientError):
        client.request('openDevice',{'email':'contact@example.org'})
    restored=FeedbackClient(tmp_path)
    def succeed(path,body,**kwargs):
        attempts.append(body.copy())
        return dict(token=body['deviceToken'],email=body['email'],owner='opaque-test-owner',emailVerified=False)
    restored.call=succeed
    status=restored.request('openDevice',{'email':'contact@example.org'})
    assert status['connected'] and not status['verified'] and 'token' not in status
    assert attempts[0]==attempts[1]
    assert attempts[0]['deviceToken'].encode() not in restored.path.read_bytes()


@pytest.mark.parametrize('status',[404,502,503,504])
def test_web_operations_missing_route_has_specific_message(tmp_path,status):
    from io import BytesIO
    from urllib.error import HTTPError
    client=FeedbackClient(tmp_path,admin=True)
    def unavailable(request,timeout):raise HTTPError(request.full_url,status,'synthetic',{},BytesIO(b'<html>unavailable</html>'))
    client.opener.open=unavailable
    with pytest.raises(FeedbackClientError,match='网页版管理暂未连接'):
        client.request('webStatus',{})
    with pytest.raises(FeedbackClientError,match='反馈服务暂时不可用'):
        client.request('conversations',{})
