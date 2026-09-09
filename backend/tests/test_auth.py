"""P05 HTTP security checks with an explicitly non-persistent test double."""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from ptb_api.auth import AuthSettings, create_account_router, hash_token
from ptb_api.main import create_app
from account_double import MemoryAccountStore

ORIGIN = 'https://research.test'

@pytest.fixture
def setup():
    store = MemoryAccountStore()
    store.create_user('alice', 'a-correct-long-password')
    store.create_user('bob', 'b-correct-long-password')
    settings = AuthSettings(origin=ORIGIN, signing_key='test-only-' * 8)
    app = create_app(account_store=store, auth_settings=settings)
    return store, app


def login(client, username='alice', password='a-correct-long-password'):
    csrf = client.get('/api/v1/auth/challenge').json()['csrf_token']
    return client.post('/api/v1/auth/login', json={'username': username, 'password': password},
                       headers={'Origin': ORIGIN, 'X-CSRF-Token': csrf})


def test_login_rotates_and_logout_revokes(setup):
    store, app = setup
    with TestClient(app, base_url=ORIGIN) as client:
        response = login(client)
        assert response.status_code == 200
        cookie = client.cookies.get('__Host-ptb-session')
        assert cookie not in str(store.sessions)
        assert hash_token(cookie) in store.sessions
        header = response.headers['set-cookie']
        assert 'HttpOnly' in header and 'Secure' in header and 'SameSite=strict' in header
        csrf = client.get('/api/v1/auth/me').json()['csrf_token']
        assert client.post('/api/v1/auth/logout', headers={'Origin': ORIGIN, 'X-CSRF-Token': csrf}).status_code == 204
        client.cookies.set('__Host-ptb-session', cookie)
        assert client.get('/api/v1/auth/me').status_code == 401


def test_csrf_origin_input_redaction_and_rate_limit(setup):
    _, app = setup
    with TestClient(app, base_url=ORIGIN) as client:
        assert client.post('/api/v1/auth/login', json={'username':'alice','password':'secret'}).status_code == 403
        token = client.get('/api/v1/auth/challenge').json()['csrf_token']
        headers = {'Origin': ORIGIN, 'X-CSRF-Token': token}
        bad = client.post('/api/v1/auth/login', json={'username':'alice','password':'private-password','owner_id':'bob'},headers=headers)
        assert bad.status_code == 422 and 'private-password' not in bad.text
        for _ in range(10):
            assert login(client,password='not-the-password').status_code == 401
        assert login(client).status_code == 429
        assert login(client, 'missing', 'also-wrong').status_code == 401


def test_session_expiry_and_disabled_user(setup):
    store, app = setup
    with TestClient(app, base_url=ORIGIN) as client:
        assert login(client).status_code == 200
        session = store.sessions[hash_token(client.cookies.get('__Host-ptb-session'))]
        session['expires_at'] = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert client.get('/api/v1/auth/me').status_code == 401
        assert login(client).status_code == 200
        store.users['alice']['active'] = False
        assert client.get('/api/v1/auth/me').status_code == 401


def test_only_explicit_loopback_allows_insecure_cookie():
    with pytest.raises(ValueError):
        AuthSettings(origin='http://research.test', signing_key='x'*64, allow_insecure_loopback=True)
    with pytest.raises(ValueError):
        AuthSettings(origin='https://research.test/path', signing_key='x'*64)
    assert not AuthSettings(origin='http://127.0.0.1:8888', signing_key='x'*64, allow_insecure_loopback=True).secure


def test_rotated_cookie_csrf_and_bound_input(setup):
    store, app = setup
    with TestClient(app,base_url=ORIGIN) as client:
        first=login(client)
        old=client.cookies.get('__Host-ptb-session')
        old_csrf=first.json()['csrf_token']
        assert login(client).status_code==200
        assert store.sessions[hash_token(old)]['revoked']
        assert client.cookies.get('__Host-ptb-session')!=old
        assert client.post('/api/v1/auth/logout',headers={'Origin':ORIGIN,'X-CSRF-Token':old_csrf}).status_code==403
        assert client.get('/api/v1/projects',headers={'X-PTB-Account':str(uuid4())}).status_code==409
        assert client.get('/api/v1/auth/me',headers={'Host':'evil.test'}).status_code==403
        secret='not-a-real-password'*2000
        response=client.post('/api/v1/auth/login',content=secret,headers={'Content-Type':'application/json'})
        assert response.status_code==413 and secret not in response.text
        challenge=client.get('/api/v1/auth/challenge').json()['csrf_token']
        assert client.post('/api/v1/auth/login',json={'username':'alice','password':'wrong'},headers={'Origin':ORIGIN,'X-CSRF-Token':challenge+'tamper'}).status_code==403
