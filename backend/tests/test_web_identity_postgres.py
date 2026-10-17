"""Approved fresh PostgreSQL only; capture messages locally, never contact SMTP."""
from concurrent.futures import ThreadPoolExecutor
from secrets import token_urlsafe
from uuid import uuid4
import re
import pytest
from fastapi.testclient import TestClient
from ptb_api.account_store import hash_token, PASSWORDS
from ptb_api.auth import AuthSettings
from ptb_api.main import create_app
from ptb_api.registration import RegistrationService
from ptb_api.registration_store import PostgresRegistrationStore
from test_p07_policy_postgres import legacy, MIGRATIONS
from web_postgres_support import cluster

ORIGIN = 'https://web-identity.test'


class MailCapture:
    def __init__(self):
        self.messages=[]

    def send(self, recipient, subject, body, key):
        self.messages.append((recipient, subject, body))

    def token(self):
        return re.search(r'#(?:verify|reset)=([A-Za-z0-9_-]{43})', self.messages[-1][2])[1]


@pytest.fixture
def web(legacy):
    e=legacy
    e.conn.execute((MIGRATIONS/'008_web_identity.sql').read_text('utf-8'))
    store=PostgresRegistrationStore(e.dsn);store.check_schema()
    mail=MailCapture()
    service=RegistrationService(store,mail,ORIGIN)
    app=create_app(account_store=store,auth_settings=AuthSettings(origin=ORIGIN,signing_key='identity-test-'*4),registration_service=service)
    return e,store,mail,app


def post(client, route, body):
    token=client.get('/api/v1/auth/challenge').json()['csrf_token']
    return client.post('/api/v1/auth/'+route,json=body,headers={'Origin':ORIGIN,'X-CSRF-Token':token})


def signup(client,mail,username='researcher',email='researcher@example.test'):
    response=post(client,'register',dict(username=username,email=email,password='synthetic-password-only'))
    assert response.status_code==202,response.text
    assert response.json()=={'accepted':True}
    token=mail.token()
    assert token not in response.text
    assert post(client,'verify-email',dict(token=token)).status_code==204


def test_registration_requires_verification_reset_revokes_sessions_and_replay(web):
    e,store,mail,app=web
    with TestClient(app,base_url=ORIGIN) as client:
        assert client.get('/api/v1/auth/options').json()==dict(registration=True,password_recovery=True,ephemeral_workspaces=False)
        response=post(client,'register',dict(username='Researcher',email='Researcher@Example.test',password='synthetic-password-only'))
        assert response.status_code==202 and store.get_user('researcher') is None
        token=mail.token()
        row=e.conn.execute('SELECT * FROM ptb_web.email_challenges').fetchone()
        assert row['token_hash']==hash_token(token) and token not in str(row)
        assert row['password_hash']!='synthetic-password-only'
        assert post(client,'verify-email',dict(token=token)).status_code==204
        assert post(client,'verify-email',dict(token=token)).status_code==400
        assert store.get_user('researcher')['active']
        assert post(client,'login',dict(username='researcher',password='synthetic-password-only')).status_code==200
        old=client.cookies.get('__Host-ptb-session')
        assert post(client,'forgot-password',dict(email='researcher@example.test')).status_code==202
        reset=mail.token()
        assert post(client,'reset-password',dict(token=reset,password='new-synthetic-password')).status_code==204
        assert store.session(hash_token(old)) is None
        assert post(client,'reset-password',dict(token=reset,password='another-synthetic-password')).status_code==400
        assert post(client,'login',dict(username='researcher',password='synthetic-password-only')).status_code==401
        assert post(client,'login',dict(username='researcher',password='new-synthetic-password')).status_code==200


def test_unknown_duplicate_expired_and_cross_origin_are_safe(web):
    e,store,mail,app=web
    with TestClient(app,base_url=ORIGIN) as client:
        signup(client,mail)
        count=len(mail.messages)
        duplicate=post(client,'register',dict(username='researcher',email='another@example.test',password='synthetic-password-only'))
        unknown=post(client,'forgot-password',dict(email='unknown@example.test'))
        assert duplicate.status_code==unknown.status_code==202
        assert duplicate.json()==unknown.json()=={'accepted':True}
        assert len(mail.messages)==count
        assert client.post('/api/v1/auth/register',json=dict(username='other',email='other@example.test',password='synthetic-password-only')).status_code==403
        response=post(client,'register',dict(username='expiring',email='expire@example.test',password='synthetic-password-only'))
        token=mail.token()
        e.conn.execute("UPDATE ptb_web.email_challenges SET expires_at=clock_timestamp()-interval '1 second'")
        assert post(client,'verify-email',dict(token=token)).status_code==400
        assert store.get_user('expiring') is None
        bad=post(client,'register',dict(username='other',email='a\r\nBcc:x@example.test',password='private-password'))
        assert bad.status_code==422 and 'private-password' not in bad.text and 'Bcc' not in bad.text


def test_one_time_consumption_and_rate_reservations_are_atomic(web):
    e,store,mail,app=web
    token=token_urlsafe(32)
    assert store.begin_registration('racer','race@example.test',PASSWORDS.hash('synthetic-password-only'),hash_token(token))
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(lambda _:store.consume_email(hash_token(token),'register'),range(4)))==1
        assert sum(pool.map(lambda _:store.allow_email_action('register','limited@example.test','test-peer'),range(10)))==5
    assert e.conn.execute("SELECT count(*) AS n FROM ptb_accounts.users WHERE username='racer'").fetchone()['n']==1


def test_resend_invalidates_old_link_without_changing_existing_password(web):
    e,store,mail,app=web
    first,second=token_urlsafe(32),token_urlsafe(32)
    digest=PASSWORDS.hash('synthetic-password-only')
    assert store.begin_registration('resend','resend@example.test',digest,hash_token(first))
    assert store.begin_registration('resend','resend@example.test',digest,hash_token(second))
    assert not store.consume_email(hash_token(first),'register')
    assert store.consume_email(hash_token(second),'register')
    assert not store.begin_registration('resend','resend@example.test',PASSWORDS.hash('malicious-overwrite'),hash_token(token_urlsafe(32)))
    assert store.get_user('resend')['password_hash']==digest
