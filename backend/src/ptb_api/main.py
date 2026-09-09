from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
import psycopg
from .account_store import AccountStore
from .auth import AccountContext, AuthSettings, create_account_router
from .projects import create_project_router
from .account_boundary import AccountBoundary
from phonetic_core import __version__ as core_version

from . import __version__
from .models import Capabilities, Health, Viewport
from .protocol_version import API_VERSION


def create_app(mode: Literal['local', 'server'] = 'server', *, account_store: AccountStore | None = None,
               auth_settings: AuthSettings | None = None) -> FastAPI:
    if mode not in ('local', 'server'):
        raise ValueError('Unsupported service mode')
    app = FastAPI(title='PhoneticToolbox API', version=API_VERSION,
                  description='Shared API; P05 accounts require configured PostgreSQL. Scientific tasks not yet connected.')
    ctx = AccountContext(account_store, auth_settings, mode)
    app.include_router(create_account_router(ctx))
    app.include_router(create_project_router(ctx))

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc):
        # FastAPI input error details can echo submitted passwords.
        return JSONResponse({'detail': 'invalid_request'}, status_code=422)

    @app.exception_handler(psycopg.Error)
    async def database_unavailable(request: Request, exc):
        return JSONResponse({'detail':'account_service_unavailable'},status_code=503)

    @app.middleware('http')
    async def private_responses(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith(('/api/v1/auth/', '/api/v1/projects')):
            response.headers['Cache-Control'] = 'no-store'
            response.headers['Pragma'] = 'no-cache'
            response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/api/v1/health', response_model=Health, operation_id='get_health')
    def health() -> Health:
        return Health(app_version=__version__, core_version=core_version, mode=mode)

    @app.get('/api/v1/capabilities', response_model=Capabilities, operation_id='get_capabilities')
    def capabilities() -> Capabilities:
        return Capabilities(stage='P05' if account_store is not None and mode == 'server' else 'P02',
                            algorithms=[], limitations=[
            'Scientific modules pending P08', 'Accounts require configured PostgreSQL; tasks pending P06'])

    base_openapi = app.openapi

    def openapi():
        schema = base_openapi()
        # Shared process models are published without adding a fake analysis endpoint.
        shared = Viewport.model_json_schema(ref_template='#/components/schemas/{model}')
        definitions = shared.pop('$defs', {})
        schema.setdefault('components', {}).setdefault('schemas', {}).update(definitions)
        schema['components']['schemas']['Viewport'] = shared
        return schema

    app.add_middleware(AccountBoundary, origin=auth_settings.origin if auth_settings else None)
    app.openapi = openapi
    return app
