"""Opaque, revocable P05 sessions; no bearer credentials in browser storage."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from secrets import token_urlsafe
from urllib.parse import urlsplit
import hmac

from argon2.exceptions import VerificationError, InvalidHashError
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from itsdangerous import URLSafeTimedSerializer, BadSignature

from .account_models import Challenge, LoginInput, SessionView, UserView
from .account_store import AccountStore, PASSWORDS, hash_token

DUMMY_HASH = PASSWORDS.hash(token_urlsafe(32))

@dataclass(frozen=True)
class AuthSettings:
    origin: str
    signing_key: str = field(repr=False)
    allow_insecure_loopback: bool = False
    lifetime_seconds: int = 8 * 3600

    def __post_init__(self):
        url = urlsplit(self.origin)
        if (url.scheme not in ('https','http') or not url.netloc or url.username or url.password
                or url.path or url.query or url.fragment or len(self.signing_key) < 32
                or not 60 <= self.lifetime_seconds <= 86400):
            raise ValueError('Invalid account origin, key or lifetime')
        if url.scheme != 'https' and not (self.allow_insecure_loopback and url.hostname == '127.0.0.1'):
            raise ValueError('HTTP is limited to explicit loopback development')

    @property
    def secure(self):
        return self.origin.startswith('https://')

    @property
    def cookie(self):
        return '__Host-ptb-session' if self.secure else 'ptb-dev-session'

    @property
    def challenge_cookie(self):
        return '__Host-ptb-login' if self.secure else 'ptb-dev-login'

class AccountContext:
    def __init__(self, store: AccountStore | None, settings: AuthSettings | None, mode: str):
        self.store, self.settings, self.mode = store, settings, mode

    def available(self):
        if self.mode != 'server':
            raise HTTPException(404, 'account_not_available')
        if self.store is None or self.settings is None:
            raise HTTPException(503, 'account_service_unavailable')

    def origin(self, request: Request):
        self.available()
        if request.headers.getlist('origin') != [self.settings.origin]:
            raise HTTPException(403, 'origin_rejected')
        if request.headers.get('sec-fetch-site') == 'cross-site':
            raise HTTPException(403, 'origin_rejected')

    def session(self, request: Request):
        self.available()
        raw = request.cookies.get(self.settings.cookie, '')
        if not 32 <= len(raw) <= 128:
            raise HTTPException(401, 'authentication_required')
        session = self.store.session(hash_token(raw))
        if not session:
            raise HTTPException(401, 'authentication_required')
        expected = request.headers.get('x-ptb-account')
        if expected is not None and expected != session['id']:
            raise HTTPException(409, 'account_changed')
        return session

    def mutation(self, request: Request):
        self.origin(request)
        session = self.session(request)
        header = request.headers.get('x-csrf-token', '')
        if not hmac.compare_digest(header.encode('utf-8'), session['csrf_token'].encode('utf-8')):
            raise HTTPException(403, 'csrf_rejected')
        return session

    def view(self, session):
        return SessionView(user=UserView(id=str(session['id']),username=session['username']),
                           csrf_token=session['csrf_token'],expires_at=session['expires_at'])


def create_account_router(ctx: AccountContext):
    router = APIRouter(prefix='/api/v1/auth', tags=['accounts'])

    @router.get('/challenge', response_model=Challenge, operation_id='get_login_challenge')
    def challenge(response: Response):
        ctx.available()
        token = URLSafeTimedSerializer(ctx.settings.signing_key, salt='ptb-login-csrf').dumps(token_urlsafe(32))
        response.set_cookie(ctx.settings.challenge_cookie, token, max_age=600, secure=ctx.settings.secure,
                            httponly=True, samesite='strict', path='/')
        return Challenge(csrf_token=token)

    @router.post('/login', response_model=SessionView, operation_id='login_account', dependencies=[Depends(ctx.origin)])
    def login(body: LoginInput, request: Request, response: Response):
        token = request.headers.get('x-csrf-token','')
        cookie = request.cookies.get(ctx.settings.challenge_cookie,'')
        if not token or len(token) > 512 or not hmac.compare_digest(token.encode('utf-8'),cookie.encode('utf-8')):
            raise HTTPException(403, 'csrf_rejected')
        try:
            URLSafeTimedSerializer(ctx.settings.signing_key, salt='ptb-login-csrf').loads(token,max_age=600)
        except BadSignature:
            raise HTTPException(403, 'csrf_rejected') from None
        username = body.username.lower()
        ip = request.client.host if request.client else 'unknown'
        if not ctx.store.allow_login(username,ip):
            raise HTTPException(429, 'login_rate_limited',headers={'Retry-After':'900'})
        user = ctx.store.get_user(username)
        try:
            valid = PASSWORDS.verify(user['password_hash'] if user else DUMMY_HASH, body.password)
        except (VerificationError, InvalidHashError):
            valid = False
        if not valid or not user or not user['active']:
            raise HTTPException(401, 'invalid_credentials')
        raw, csrf = token_urlsafe(32), token_urlsafe(32)
        expires = datetime.now(timezone.utc)+timedelta(seconds=ctx.settings.lifetime_seconds)
        if not ctx.store.issue_session(user,hash_token(raw),csrf,expires,hash_token(request.cookies.get(ctx.settings.cookie,''))):
            raise HTTPException(401,'invalid_credentials')
        response.set_cookie(ctx.settings.cookie,raw,max_age=ctx.settings.lifetime_seconds,secure=ctx.settings.secure,
                            httponly=True,samesite='strict',path='/')
        response.delete_cookie(ctx.settings.challenge_cookie,path='/',secure=ctx.settings.secure,httponly=True,samesite='strict')
        return ctx.view({'id':user['id'],'username':user['username'],'csrf_token':csrf,'expires_at':expires})

    @router.get('/me', response_model=SessionView, operation_id='get_current_account')
    def me(session=Depends(ctx.session)):
        return ctx.view(session)

    @router.post('/logout', status_code=204, operation_id='logout_account')
    def logout(request: Request, response: Response, session=Depends(ctx.mutation)):
        ctx.store.revoke(hash_token(request.cookies.get(ctx.settings.cookie,'')))
        response.delete_cookie(ctx.settings.cookie,path='/',secure=ctx.settings.secure,httponly=True,samesite='strict')
        response.status_code=204
        return response

    return router
