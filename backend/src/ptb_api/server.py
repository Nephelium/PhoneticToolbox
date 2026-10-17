"""P05 loopback account server. Reads private config from stdin, never from URL/argv."""
import argparse
import json
import socket
import sys
import threading
from urllib.parse import urlsplit
from pathlib import Path
import uvicorn
import psycopg
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from .main import create_app
from .auth import AuthSettings
from .account_store import PostgresAccountStore
from .cli import LoopbackGuard
from ptb_worker.store import PostgresJobStore
from .storage import Storage
from ptb_worker.cleanup import run_cleanup


def account_settings(config, port):
    origin = config.get('public_origin')
    if origin is not None:
        if not isinstance(origin, str) or not origin.startswith('https://'):
            raise ValueError('Public account origin requires HTTPS')
        return AuthSettings(origin=origin, signing_key=config['signing_key'])
    return AuthSettings(origin=f'http://127.0.0.1:{port}',
                        signing_key=config['signing_key'], allow_insecure_loopback=True)


class PublicOriginGuard:
    """Nginx connects over loopback while preserving the configured HTTPS host."""
    def __init__(self, app, origin):
        self.app, self.origin, self.authority = app, origin, urlsplit(origin).netloc

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            hosts = [v.decode('latin1') for k, v in scope['headers'] if k == b'host']
            origins = [v.decode('latin1') for k, v in scope['headers'] if k == b'origin']
            if hosts != [self.authority] or (origins and origins != [self.origin]):
                await JSONResponse({'detail': 'origin_rejected'}, status_code=403,
                                   headers={'Cache-Control': 'no-store'})(scope, receive, send)
                return
        await self.app(scope, receive, send)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=5175)
    parser.add_argument('--frontend',type=Path,required=True)
    parser.add_argument('--managed',action='store_true',help='Exit when the owning launcher closes stdin')
    parser.add_argument('--config',type=Path,help='Owned private configuration for an isolated loopback instance')
    args=parser.parse_args()
    static=args.frontend.resolve()
    if not (static/'index.html').is_file():
        parser.error('Build the frontend before starting the account preview')
    if args.config:
        from .storage import no_links
        import stat,os
        no_links(args.config)
        info=args.config.stat()
        if os.name=='posix' and (info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600):
            raise ValueError('Private loopback configuration must be owned and mode 0600')
        config=json.loads(args.config.read_text('utf-8'))
    else:
        config=json.loads(sys.stdin.readline())
    entry=config.get('entry_path','/server/')
    if entry not in ('/server/','/app/'):raise ValueError('Unsupported preview entry')
    if config.get('enable_registration') and not config.get('enable_web_workspaces'):
        raise ValueError('Public registration requires ephemeral web workspaces')
    settings=account_settings(config,args.port)
    store=PostgresAccountStore(config['dsn'])
    # Read only: neither construction nor startup performs a migration.
    store.check_schema()
    jobs=PostgresJobStore(config['dsn'],max_running=1 if config.get('enable_web_workspaces') else 2) if config.get('enable_jobs',False) else None
    if jobs:jobs.check_schema()
    storage=Storage(config['dsn'],config['storage_root'],min_free_bytes=config.get('min_free_bytes',1_000_000_000)) if config.get('storage_root') else None
    workspaces=None
    if config.get('enable_web_workspaces', False):
        if storage is None or jobs is None:
            raise ValueError('Web workspaces require PostgreSQL jobs and private storage')
        from .workspaces import WorkspaceService
        workspaces=WorkspaceService(storage)
        workspaces.activate()
    if storage:
        if jobs and config.get('enable_file_jobs',False):
            from ptb_worker.files import FilePipeline
            if config.get('enable_acoustic_batches'):
                from ptb_worker.acoustic_files import AcousticFiles
                from ptb_worker.acoustic_batches import AcousticBatches
                files=AcousticFiles(jobs,storage);AcousticBatches(jobs,files)
                files.reaper_binary=config.get('reaper_binary')
            else:FilePipeline(jobs,storage)
        storage.recover()
    registration = None
    if config.get('enable_registration', False):
        from .registration import RegistrationService
        from .registration_store import PostgresRegistrationStore
        from .feedback_mail import SMTPMailer
        identity_store = PostgresRegistrationStore(config['dsn'])
        identity_store.check_schema()
        registration = RegistrationService(identity_store, SMTPMailer(config['smtp']), settings.origin,entry_path=entry)
    from .web_feedback import WebFeedbackGateway
    web_feedback=WebFeedbackGateway(config['web_admin_key']) if config.get('web_admin_key') else None
    app=create_app(account_store=store,auth_settings=settings,job_store=jobs,storage=storage,
                   registration_service=registration,workspace_service=workspaces,web_admin_key=config.get('web_admin_key'),vocal_resources=config.get('vocal_resources'),web_feedback_gateway=web_feedback)
    app.mount(entry.rstrip('/'),StaticFiles(directory=static,html=True),name='account-ui')
    guarded=PublicOriginGuard(app,settings.origin) if config.get('public_origin') else LoopbackGuard(app,f'127.0.0.1:{args.port}',None)
    server=uvicorn.Server(uvicorn.Config(guarded,host='127.0.0.1',port=args.port,access_log=False,log_level='warning',
        proxy_headers=bool(config.get('public_origin')),forwarded_allow_ips='127.0.0.1'))
    if args.managed:
        def owner_closed():sys.stdin.readline();server.should_exit=True
        threading.Thread(target=owner_closed,daemon=True).start()
    cleanup_stop=threading.Event()
    cleaner=threading.Thread(target=run_cleanup,args=(storage,cleanup_stop),daemon=True) if storage else None
    worker=None
    if config.get('enable_worker',False):
        if jobs is None or jobs.files is None:
            raise ValueError('Preview worker requires configured file jobs')
        from uuid import uuid4
        from ptb_worker.executor import run_worker
        worker=threading.Thread(target=run_worker,args=(jobs,cleanup_stop,'preview-'+uuid4().hex),daemon=True)
        worker.start()
    if cleaner:
        cleaner.start()
    try:
        server.run()
    finally:
        cleanup_stop.set()
        if worker:
            worker.join(timeout=30)
        if cleaner:
            cleaner.join(timeout=15)
        if worker and worker.is_alive():
            raise RuntimeError('Preview worker did not stop within its shutdown budget')

if __name__=='__main__':
    try:
        main()
    except psycopg.Error:
        print('Account database unavailable; verify private connection settings and approved schema.',file=sys.stderr)
        raise SystemExit(1) from None
