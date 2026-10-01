from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
import psycopg
import sqlite3
from .jobs import create_job_router
from ptb_worker.store import JobError, JobStore
from .account_store import AccountStore
from .auth import AccountContext, AuthSettings, create_account_router
from .projects import create_project_router
from .assets import create_storage_router
from .preview import create_preview_router
from ptb_worker.spectrogram_preview import PreviewError
from .quota import StorageError
from .account_boundary import AccountBoundary
from phonetic_core import __version__ as core_version

from . import __version__
from .models import Capabilities, Health, Viewport
from .protocol_version import API_VERSION
from .acoustic_models import ACOUSTIC_SCHEMAS
from .egg_models import EggPreviewData, EggInverseData
from .lpc_models import LpcSpectrumData
from .job_models import ResultManifestEnvelope


def create_app(mode: Literal['local', 'server'] = 'server', *, account_store: AccountStore | None = None,
               auth_settings: AuthSettings | None = None, job_store: JobStore | None = None,
               local_token: str | None = None, local_origin: str | None = None, storage=None) -> FastAPI:
    if mode not in ('local', 'server'):
        raise ValueError('Unsupported service mode')
    from contextlib import asynccontextmanager
    from ptb_worker.egg_interactive import InteractivePreview
    egg_preview = InteractivePreview(memory_bytes=3_000_000_000 if mode == 'local' else 1_000_000_000)
    from ptb_worker.spectrogram_session import SpectrogramSession
    spectrogram_session = SpectrogramSession() if mode == 'local' else None
    @asynccontextmanager
    async def lifespan(app):
        try: yield
        finally:
            egg_preview.close()
            if spectrogram_session is not None:spectrogram_session.close()
    app = FastAPI(title='PhoneticToolbox API', version=API_VERSION, lifespan=lifespan,
                  description='Shared API; accounts require configured PostgreSQL. M01 batches require an explicitly configured durable worker and resource store.')
    ctx = AccountContext(account_store, auth_settings, mode)
    app.include_router(create_account_router(ctx))
    app.include_router(create_project_router(ctx))
    app.include_router(create_job_router(ctx,job_store,local_token=local_token,local_origin=local_origin))
    app.include_router(create_storage_router(ctx, storage, egg_preview=egg_preview))
    app.include_router(create_preview_router(mode,local_token,local_origin,egg_preview=egg_preview,spectrogram_session=spectrogram_session))

    @app.exception_handler(PreviewError)
    async def preview_error(request: Request,exc):
        return JSONResponse({'detail':exc.code},status_code=exc.status)

    @app.exception_handler(StorageError)
    async def storage_error(request: Request, exc):
        return JSONResponse({'detail': exc.code}, status_code=exc.status)

    @app.exception_handler(OSError)
    async def file_unavailable(request: Request, exc):
        return JSONResponse({'detail': 'storage_service_unavailable'}, status_code=503)

    @app.exception_handler(JobError)
    async def job_error(request: Request, exc):
        return JSONResponse({'detail':exc.code},status_code=exc.status)

    @app.exception_handler(sqlite3.Error)
    async def local_storage_unavailable(request: Request, exc):
        return JSONResponse({'detail':'task_service_unavailable'},status_code=503)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc):
        # FastAPI input error details can echo submitted passwords.
        return JSONResponse({'detail': 'invalid_request'}, status_code=422)

    @app.exception_handler(psycopg.Error)
    async def database_unavailable(request: Request, exc):
        path = request.url.path
        code = ('storage_service_unavailable' if path.startswith(('/api/v1/storage', '/api/v1/assets', '/api/v1/uploads'))
                else 'task_service_unavailable' if path.startswith('/api/v1/jobs') else 'account_service_unavailable')
        return JSONResponse({'detail': code}, status_code=503)

    @app.middleware('http')
    async def private_responses(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith(('/api/v1/auth/', '/api/v1/projects', '/api/v1/jobs', '/api/v1/storage', '/api/v1/assets', '/api/v1/uploads','/api/v1/preview')):
            response.headers['Cache-Control'] = 'no-store'
            response.headers['Pragma'] = 'no-cache'
            response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/api/v1/health', response_model=Health, operation_id='get_health')
    def health() -> Health:
        return Health(app_version=__version__, core_version=core_version, mode=mode)

    @app.get('/api/v1/capabilities', response_model=Capabilities, operation_id='get_capabilities')
    def capabilities(request: Request) -> Capabilities:
        from ptb_worker.native.capabilities import m01_capability, linux_capabilities, platform_name
        storage_ready = mode == 'server' and storage is not None and getattr(storage, 'ready', False)
        file_jobs = storage_ready and job_store is not None and getattr(job_store,'files',None) is not None
        acoustic_ready, acoustic_reason = m01_capability(job_store)
        science_operations = ['acoustic_analysis','textgrid_segment'] if acoustic_ready else []
        algorithms = ['M01'] if acoustic_ready else []
        from ptb_worker.m05_task import catalog as m05_catalog
        if job_store is not None and m05_catalog(mode=='local')['available']:
            science_operations.append('lip_analysis'); algorithms.append('M05')
        from ptb_worker.m11_task import catalog as m11_catalog
        if m11_catalog(local=mode=='local')['execution_available']:
            science_operations.append('mfa_alignment'); algorithms.append('M11')
        from ptb_worker.m07_task import capability as m07_capability
        if m07_capability(job_store):
            science_operations.append('phonation_synthesis'); algorithms.append('M07')
        from ptb_worker.m06_task import capability as m06_capability
        if m06_capability(job_store):
            science_operations.append('speech_synthesis'); algorithms.append('M06')
        from ptb_worker.m08_task import windows_capability
        if windows_capability(job_store):
            science_operations.append('pitch_manipulation'); algorithms.append('M08')
        from ptb_worker.m14_task import capability as m14_capability
        if m14_capability() and job_store is not None and getattr(job_store,'files',None) is not None:
            science_operations.append('phonology_induction'); algorithms.append('M14')
        if platform_name() == 'linux':
            science_operations, reasons = linux_capabilities(job_store)
            algorithms = [module for op,module in [('lpc_analysis','M04'),('egg_analysis','M03'),('acoustic_analysis','M01')] if op in science_operations]
            acoustic_reason = reasons[0] if reasons else None
        from ptb_worker.resource_profiles import selected_profile
        from ptb_worker.io.limits import FormatError
        try: archive_ready = not selected_profile().shared_admission
        except FormatError: archive_ready = False
        operations = (['pipeline_check'] + (['storage_check'] + (['archive_zip','extract_zip'] if archive_ready else []) if file_jobs else []) + science_operations) if job_store is not None else []
        # Set only by the trusted ASGI deployment boundary, never by HTTP input.
        allowed = request.scope.get('ptb.allowed_operations')
        if allowed is not None:
            operations = [operation for operation in operations if operation in allowed]
            module_ops = {'M01':{'acoustic_analysis','textgrid_segment'},'M03':{'egg_analysis'},
                'M04':{'lpc_analysis'},'M05':{'lip_analysis'},'M06':{'speech_synthesis'},
                'M07':{'phonation_synthesis'},'M08':{'pitch_manipulation'},
                'M11':{'mfa_alignment'},'M14':{'phonology_induction'}}
            algorithms = [module for module in algorithms if module_ops.get(module,set()).intersection(operations)]
        storage_operations = ['upload','download','delete'] if storage_ready else []
        if request.scope.get('ptb.storage_readonly'):
            storage_operations = [op for op in storage_operations if op=='download']
        return Capabilities(stage='P07' if storage_ready else ('P06' if job_store is not None else ('P05' if account_store is not None and mode == 'server' else 'P02')),
                            algorithms=algorithms, task_operations=operations,
                            storage_operations=storage_operations, limitations=[
            'Linux science requires host-selected runtime hashes and completed task validation receipts; Windows M01 requires its registered native resource and matching packages',
            'Capabilities cover configured execution paths only; whole-host concurrency, remote execution and M09 Linux remain unverified',
            'Storage checks generate engineering fixtures, not scientific analysis results'] + ([acoustic_reason] if acoustic_reason else []))

    base_openapi = app.openapi

    def openapi():
        schema = base_openapi()
        # Shared process models are published without adding a fake analysis endpoint.
        for model in (Viewport,*ACOUSTIC_SCHEMAS,ResultManifestEnvelope,EggPreviewData,EggInverseData,LpcSpectrumData):
            shared = model.model_json_schema(ref_template='#/components/schemas/{model}')
            definitions = shared.pop('$defs', {})
            schema.setdefault('components', {}).setdefault('schemas', {}).update(definitions)
            schema['components']['schemas'][model.__name__] = shared
        return schema

    app.add_middleware(AccountBoundary, origin=auth_settings.origin if auth_settings else None)
    app.openapi = openapi
    return app
