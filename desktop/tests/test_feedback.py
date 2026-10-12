from pathlib import Path
import pytest
from ptb_desktop.feedback import FeedbackClient,FeedbackClientError,protected,valid_base
from ptb_api.feedback_mail import settings_from_script


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
