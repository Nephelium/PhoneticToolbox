"""Actual account-language UI across owned Chrome/Edge and fresh PostgreSQL."""
import socket
import subprocess
import threading
import time
from pathlib import Path

import uvicorn
from starlette.staticfiles import StaticFiles
from ptb_api.account_store import PASSWORDS
from ptb_api.auth import AuthSettings
from ptb_api.main import create_app
from ptb_api.preferences import list_preferences
from test_account_preferences_postgres import web
from test_p07_policy_postgres import legacy
from web_postgres_support import cluster, artifacts
from web_candidate_support import prepare_frontend, verify_frontend_unchanged


def test_account_language_and_live_workspace_in_independent_browser_profiles(web):
    e=web;root=Path(__file__).resolve().parents[2]
    e.conn.execute('UPDATE ptb_accounts.users SET password_hash=%s WHERE id=%s',(PASSWORDS.hash('synthetic-password-only'),e.owner))
    e.accounts.create_user('language-other','synthetic-password-only')
    with artifacts.verification_run('app-i18n-web','account-browser','Real independent Chrome/Edge profiles, fresh owned PG, actual HTTP; language/account/WAV/parameter and nine-width checks; no SMTP or public services') as (out,scratch):
        frontend,frontend_identity=prepare_frontend(root,scratch,out)
        sock=socket.socket();sock.bind(('127.0.0.1',0));origin=f'http://127.0.0.1:{sock.getsockname()[1]}'
        app=create_app(account_store=e.accounts,auth_settings=AuthSettings(origin=origin,signing_key='isolated-app-language-'*3,allow_insecure_loopback=True),job_store=e.jobs,storage=e.storage,workspace_service=e.workspaces,vocal_resources=root/'resources/vocal_tract/native')
        app.mount('/server',StaticFiles(directory=frontend,html=True))
        server=uvicorn.Server(uvicorn.Config(app,access_log=False,log_level='warning'))
        thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);thread.start()
        try:
            deadline=time.monotonic()+15
            while not server.started and time.monotonic()<deadline:time.sleep(.02)
            assert server.started
            with (out/'browser.log').open('w',encoding='utf-8') as log:
                subprocess.run(['node',str(root/'tests/e2e/app-i18n-web.cjs'),origin,str(out),str(scratch)],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
            saved={row['module']:row['values'] for row in list_preferences(e.accounts,e.owner)}
            assert saved['APP']['uiLocale']=='ja' and str(saved['M04']['draft']['order'])=='72'
        finally:
            server.should_exit=True;thread.join(15);sock.close();assert not thread.is_alive()
            verify_frontend_unchanged(frontend,frontend_identity,out)
