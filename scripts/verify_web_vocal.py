"""Owned loopback HTTP/browser QA with real VTL and explicit in-memory test accounts."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from workbench_source import configure
from verification_artifacts import verification_run,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resources',type=Path,default=ROOT/'output/validation/web-m10/native');parser.add_argument('--frontend',type=Path,default=ROOT/'frontend/dist');parser.add_argument('--port',type=int,default=18779)
    parser.add_argument('--origin');parser.add_argument('--account-file',type=Path);parser.add_argument('--boundaries',action='store_true')
    args=parser.parse_args()
    if args.origin:
        if not args.account_file:parser.error('--account-file required for existing private server')
        with verification_run('web-m10','linux-browser','Isolated Chrome through owned SSH tunnel, actual Linux VTL and existing PostgreSQL account/workspace; logout only this test session') as (out,scratch):
            env={**os.environ,'PTB_VOCAL_QA_ACCOUNT':str(args.account_file.resolve())}
            script='web-vocal-boundaries.cjs' if args.boundaries else 'web-vocal.cjs'
            result=subprocess.run(['node',str(ROOT/'tests/e2e'/script),args.origin,str(out),str(scratch)],cwd=ROOT,env=env)
            if result.returncode:raise RuntimeError('Private Linux browser acceptance failed')
            write_json(out/'scope.json',dict(http=args.origin,science='dedicated Linux VTL candidate',accounts='existing PostgreSQL identity/workspace',production=False))
        return
    configure(ROOT)
    sys.path.insert(0,str(ROOT/'tests/support'))
    import uvicorn
    from starlette.staticfiles import StaticFiles
    from ptb_api.main import create_app
    from ptb_api.auth import AuthSettings
    from account_double import MemoryAccountStore
    store=MemoryAccountStore();store.create_user('vocal-qa','vocal-qa-local-password');origin=f'http://127.0.0.1:{args.port}'
    app=create_app(account_store=store,auth_settings=AuthSettings(origin=origin,signing_key='vocal-local-test-'*6,allow_insecure_loopback=True),vocal_resources=args.resources)
    app.mount('/app',StaticFiles(directory=args.frontend,html=True))
    with verification_run('web-m10','browser','Real VTL HTTP server with synthetic MemoryAccountStore, Chrome isolated profile, 33/65 geometry, browser PCM, keyframes, documents and MP4') as (out,scratch):
        server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=args.port,access_log=False,log_level='warning'))
        thread=threading.Thread(target=server.run);thread.start()
        try:
            for _ in range(100):
                try:
                    urllib.request.urlopen(origin+'/api/v1/health',timeout=.5).close();break
                except OSError:time.sleep(.1)
            else:raise RuntimeError('Owned HTTP server did not start')
            result=subprocess.run(['node',str(ROOT/'tests/e2e/web-vocal.cjs'),origin,str(out),str(scratch)],cwd=ROOT)
            if result.returncode:raise RuntimeError('Browser acceptance failed')
            write_json(out/'scope.json',dict(http='FastAPI real server',science='VTL current Windows candidate',accounts='MemoryAccountStore test double; no PostgreSQL/deployment claim'))
        finally:
            server.should_exit=True;thread.join(timeout=15)
            if thread.is_alive():raise RuntimeError('Owned HTTP server failed to close')


if __name__=='__main__':main()
