"""Private M10 candidate entry. Uses existing identity/workspace schema, no migration.

Does not recover storage, run a queue worker or clean other projects. Only the
candidate's VTL sessions and its authenticated workspace lifecycle are served.
"""
import argparse
import json
from pathlib import Path
import sys
import uvicorn
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--port',type=int,default=18779)
    args=parser.parse_args();root=args.root.resolve()
    if sys.platform!='linux' or not root.name.startswith('vocal-candidate-'):raise ValueError('Dedicated Linux candidate required')
    from ptb_api.main import create_app
    from ptb_api.auth import AuthSettings
    from ptb_api.account_store import PostgresAccountStore
    from ptb_api.storage import Storage
    from ptb_api.workspaces import WorkspaceService
    from ptb_worker.store import PostgresJobStore
    config=json.loads(args.config.read_text('utf-8'))
    accounts=PostgresAccountStore(config['dsn']);accounts.check_schema()
    storage=Storage(config['dsn'],config['storage_root']);workspaces=WorkspaceService(storage);workspaces.activate()
    jobs=PostgresJobStore(config['dsn'],max_running=1);jobs.check_schema()
    app=create_app(account_store=accounts,auth_settings=AuthSettings(origin=f'http://127.0.0.1:{args.port}',signing_key=config['signing_key'],allow_insecure_loopback=True),
        storage=storage,workspace_service=workspaces,job_store=jobs,vocal_resources=root/'candidate-vocal-resources')
    @app.middleware('http')
    async def candidate_scope(request,call_next):
        path=request.url.path
        if path.startswith('/api/v1/') and not path.startswith(('/api/v1/auth/','/api/v1/workspaces/','/api/v1/projects','/api/v1/vocal/','/api/v1/health','/api/v1/capabilities')):
            return JSONResponse({'detail':'vocal_candidate_only'},status_code=503,headers={'Cache-Control':'no-store'})
        return await call_next(request)
    app.mount('/app',StaticFiles(directory=root/'frontend/dist',html=True),name='vocal-candidate-ui')
    uvicorn.run(app,host='127.0.0.1',port=args.port,access_log=False,log_level='warning',proxy_headers=False)


if __name__=='__main__':main()
