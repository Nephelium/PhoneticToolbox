"""P05 loopback account server. Reads private config from stdin, never from URL/argv."""
import argparse
import json
import socket
import sys
from pathlib import Path
import uvicorn
import psycopg
from starlette.staticfiles import StaticFiles
from .main import create_app
from .auth import AuthSettings
from .account_store import PostgresAccountStore
from .cli import LoopbackGuard


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=5175)
    parser.add_argument('--frontend',type=Path,required=True)
    args=parser.parse_args()
    static=args.frontend.resolve()
    if not (static/'index.html').is_file():
        parser.error('Build the frontend before starting the account preview')
    config=json.loads(sys.stdin.readline())
    settings=AuthSettings(origin=f'http://127.0.0.1:{args.port}',signing_key=config['signing_key'],allow_insecure_loopback=True)
    store=PostgresAccountStore(config['dsn'])
    # Read only: neither construction nor startup performs a migration.
    store.check_schema()
    app=create_app(account_store=store,auth_settings=settings)
    app.mount('/server',StaticFiles(directory=static,html=True),name='account-ui')
    guarded=LoopbackGuard(app,f'127.0.0.1:{args.port}',None)
    uvicorn.run(guarded,host='127.0.0.1',port=args.port,access_log=False,log_level='warning',proxy_headers=False)

if __name__=='__main__':
    try:
        main()
    except psycopg.Error:
        print('Account database unavailable; verify private connection settings and approved schema.',file=sys.stderr)
        raise SystemExit(1) from None
