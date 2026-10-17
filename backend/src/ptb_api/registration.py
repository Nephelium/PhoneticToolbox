"""Opt-in server registration and password recovery; desktop has no mail path."""
import logging
from secrets import token_urlsafe
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from .account_store import PASSWORDS, hash_token
from .registration_models import (EmailAccepted, EmailInput, PublicAccountOptions,
    RegisterInput, ResetPasswordInput, VerifyEmailInput)

LOG = logging.getLogger(__name__)


class RegistrationService:
    def __init__(self, store, mailer, origin, entry_path='/server/'):
        self.store, self.mailer, self.origin = store, mailer, origin
        if entry_path not in ('/server/','/app/'):raise ValueError('Unsupported web entry')
        self.entry_path=entry_path

    def reserve(self, action, target, request):
        # Only the ASGI peer is trusted. Proxy trust belongs to the deployment.
        ip = request.client.host if request.client else 'unknown'
        if not self.store.allow_email_action(action, target, ip):
            raise HTTPException(429, 'email_rate_limited', headers={'Retry-After': '900'})

    def deliver(self, email, token, purpose):
        action = '验证邮箱' if purpose == 'register' else '重设密码'
        fragment = 'verify' if purpose == 'register' else 'reset'
        minutes = 30 if purpose == 'register' else 15
        body = (f'请打开以下链接{action}，链接在{minutes}分钟内有效且只能使用一次。\n\n'
                f'{self.origin}{self.entry_path}#{fragment}={token}\n\n'
                '如果这不是你发起的操作，请忽略此邮件。')
        try:
            self.mailer.send(email, f'PhoneticToolbox · {action}', body, token_urlsafe(16))
        except Exception:
            # Resend issues a new challenge. Do not log address, token or SMTP exception.
            LOG.warning('web_identity_mail_delivery_failed')


def create_registration_router(ctx, service: RegistrationService | None):
    router = APIRouter(prefix='/api/v1/auth', tags=['accounts'])

    def available():
        ctx.available()
        if service is None:
            raise HTTPException(503, 'registration_unavailable')

    def protected(request: Request):
        available()
        ctx.origin(request)
        ctx.challenge(request)

    @router.get('/options', response_model=PublicAccountOptions, operation_id='public_account_options')
    def options():
        ctx.available()
        return PublicAccountOptions(registration=service is not None, password_recovery=service is not None,
            ephemeral_workspaces=ctx.workspaces is not None)

    @router.post('/register', response_model=EmailAccepted, status_code=202,
                 operation_id='request_registration', dependencies=[Depends(protected)])
    def register(body: RegisterInput, request: Request, background: BackgroundTasks):
        service.reserve('register', body.email, request)
        token = token_urlsafe(32)
        digest = PASSWORDS.hash(body.password)
        if service.store.begin_registration(body.username.lower(), body.email, digest, hash_token(token)):
            background.add_task(service.deliver, body.email, token, 'register')
        return EmailAccepted()

    @router.post('/verify-email', status_code=204, operation_id='verify_registration', dependencies=[Depends(protected)])
    def verify(body: VerifyEmailInput, request: Request):
        service.reserve('consume', hash_token(body.token), request)
        if not service.store.consume_email(hash_token(body.token), 'register'):
            raise HTTPException(400, 'email_link_invalid')
        return Response(status_code=204)

    @router.post('/forgot-password', response_model=EmailAccepted, status_code=202,
                 operation_id='request_password_recovery', dependencies=[Depends(protected)])
    def forgot(body: EmailInput, request: Request, background: BackgroundTasks):
        service.reserve('reset', body.email, request)
        token = token_urlsafe(32)
        if service.store.begin_recovery(body.email, hash_token(token)):
            background.add_task(service.deliver, body.email, token, 'reset')
        return EmailAccepted()

    @router.post('/reset-password', status_code=204, operation_id='reset_password', dependencies=[Depends(protected)])
    def reset(body: ResetPasswordInput, request: Request):
        service.reserve('consume', hash_token(body.token), request)
        if not service.store.consume_email(hash_token(body.token), 'reset', PASSWORDS.hash(body.password)):
            raise HTTPException(400, 'email_link_invalid')
        return Response(status_code=204)

    return router
