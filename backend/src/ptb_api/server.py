"""P05 loopback account server. Reads private config from stdin, never from URL/argv."""
import argparse
import json
import socket
import sys
import threading
from pathlib import Path
import uvicorn
import psycopg
from starlette.staticfiles import StaticFiles
from .main import create_app
from .auth import AuthSettings
from .account_store import PostgresAccountStore
from .cli import LoopbackGuard
from ptb_worker.store import PostgresJobStore
from .storage import Storage
from ptb_worker.cleanup import run_cleanup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=5175)
    parser.add_argument('--frontend',type=Path,required=True)
    parser.add_argument('--managed',action='store_true',help='Exit when the owning launcher closes stdin')
    args=parser.parse_args()
    static=args.frontend.resolve()
    if not (static/'index.html').is_file():
        parser.error('Build the frontend before starting the account preview')
    config=json.loads(sys.stdin.readline())
    settings=AuthSettings(origin=f'http://127.0.0.1:{args.port}',signing_key=config['signing_key'],allow_insecure_loopback=True)
    store=PostgresAccountStore(config['dsn'])
    # Read only: neither construction nor startup performs a migration.
    store.check_schema()
    jobs=PostgresJobStore(config['dsn']) if config.get('enable_jobs',False) else None
    if jobs:jobs.check_schema()
    storage=Storage(config['dsn'],config['storage_root']) if config.get('storage_root') else None
    if storage:
        storage.recover()
    app=create_app(account_store=store,auth_settings=settings,job_store=jobs,storage=storage)
    app.mount('/server',StaticFiles(directory=static,html=True),name='account-ui')
    guarded=LoopbackGuard(app,f'127.0.0.1:{args.port}',None)
    server=uvicorn.Server(uvicorn.Config(guarded,host='127.0.0.1',port=args.port,access_log=False,log_level='warning',proxy_headers=False))
    if args.managed:
        def owner_closed():sys.stdin.readline();server.should_exit=True
        threading.Thread(target=owner_closed,daemon=True).start()
    cleanup_stop=threading.Event()
    cleaner=threading.Thread(target=run_cleanup,args=(storage,cleanup_stop),daemon=True) if storage else None
    if cleaner:
        cleaner.start()
    try:
        server.run()
    finally:
        cleanup_stop.set()
        if cleaner:
            cleaner.join(timeout=15)

if __name__=='__main__':
    try:
        main()
    except psycopg.Error:
        print('Account database unavailable; verify private connection settings and approved schema.',file=sys.stderr)
        raise SystemExit(1) from None
