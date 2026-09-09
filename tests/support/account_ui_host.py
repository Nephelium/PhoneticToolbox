"""LOCAL TEST DOUBLE ONLY. No PostgreSQL or persistence claim. Stop after UI checks."""
import argparse
import secrets
from pathlib import Path
import uvicorn
from starlette.staticfiles import StaticFiles
from ptb_api.main import create_app
from ptb_api.auth import AuthSettings
from ptb_api.cli import LoopbackGuard
from account_double import MemoryAccountStore

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=5176)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    store=MemoryAccountStore()
    for username in ('alice','bob'):
        store.create_user(username,'P05-ui-test-password')
    app=create_app(account_store=store,auth_settings=AuthSettings(origin=f'http://127.0.0.1:{args.port}',signing_key=secrets.token_urlsafe(48),allow_insecure_loopback=True))
    app.mount('/server',StaticFiles(directory=root/'frontend/dist',html=True))
    uvicorn.run(LoopbackGuard(app,f'127.0.0.1:{args.port}',None),host='127.0.0.1',port=args.port,
                access_log=False,log_level='warning',proxy_headers=False)
