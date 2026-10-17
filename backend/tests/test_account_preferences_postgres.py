"""Account settings against an explicitly new, owned PostgreSQL cluster only."""
from concurrent.futures import ThreadPoolExecutor
import json
import pytest
from fastapi.testclient import TestClient
from ptb_api.account_store import PostgresAccountStore
from ptb_api.preferences import FIELDS
from test_web_workspaces_postgres import device, asset, expire, ORIGIN
from test_p07_policy_postgres import legacy, migrate, MIGRATIONS
from web_postgres_support import cluster
from ptb_api.workspaces import WorkspaceService
from ptb_api.main import create_app
from ptb_api.auth import AuthSettings
from ptb_worker.store import PostgresJobStore
from ptb_worker.files import FilePipeline

@pytest.fixture
def web(legacy):
    e=legacy;migrate(e)
    for name in ('008_web_identity.sql','009_web_workspaces.sql','010_web_capacity.sql','011_account_preferences.sql'):
        e.conn.execute((MIGRATIONS/name).read_text('utf-8'))
    e.accounts=PostgresAccountStore(e.dsn)
    e.workspaces=WorkspaceService(e.storage);e.workspaces.activate()
    e.jobs=PostgresJobStore(e.dsn);e.files=FilePipeline(e.jobs,e.storage)
    e.app=create_app(account_store=e.accounts,auth_settings=AuthSettings(origin=ORIGIN,signing_key='settings-test-'*4),job_store=e.jobs,storage=e.storage,workspace_service=e.workspaces)
    e.clients=[]
    yield e
    for client in e.clients:client.close()

def patch(client,module,changes,revision=0):
    return client.patch('/api/v1/preferences/'+module,json={'expected_revision':revision,'changes':changes})

SAMPLES={
 'APP':{'theme':'dark','fonts':{'version':1,'zh':'SimSun','latin':'Times New Roman','mono':'JetBrains Mono','ipa':'Doulos SIL','bodySize':15,'figure':{'follow':True,'zh':'SimSun','latin':'Times New Roman','size':13}}},
 'M01':{'parameters':['F0'],'settings':{'min_f0':65},'egg':{'enabled':True,'values':{'egg_channel':0}}},
 'M02':{'groups':[{'id':1,'title':'图窗 1','parameters':['F0']}],'reaper':False},
 'M03':{'spec_vmin':-80,'batchConfig':{'keep_praat_f0':True}},'M04':{'draft':{'order':'72','amp_max_db':'40'}},
 'M05':{'fps':25,'mirror':False},'M06':{'curveOverrides':{'F0':180,'AV':None},'duration':3},
 'M07':{'count':13,'min_f0_hz':60},'M08':{'speed':'1.2','refs':[100,200]},
 'M09':{'n_iter':64,'frequency_axis':'log'},'M10':{'frame':{'view':{'mode':'overlay'},'audio':{'volume':0.4},'pose':{'params':[1,2,3]}}},
 'M11':{'config':{'beam':15,'retry_beam':60}},'M12':{'phoneSplit':'equal','showWidth':False},
 'M13':{'standard':'汉语拼音','hanziSize':36},'M14':{'options':{'skip_first_row':False}},
 'M15':{'isi':650,'validKeys':[{'key':'f','label':'左'}]},'M16':{'config':{'sample_rate':44100,'roles':['microphone','egg']},'playVolume':0.45},
 'M17':{'textSize':34,'system':'extipa'},'M18':{'paperInterval':0.25,'paperAutoReceive':False,'readerZoom':150},
}

def test_all_modules_follow_account_to_a_new_device_and_store_instance(web):
    e=web;a,_,_,_=device(e)
    assert set(SAMPLES)==set(FIELDS)
    for module,values in SAMPLES.items():
        response=patch(a,module,values);assert response.status_code==200,response.text
    b,_,_,_=device(e)
    remote=b.get('/api/v1/preferences');assert remote.status_code==200
    assert {item['module']:item['values'] for item in remote.json()['modules']}==SAMPLES
    assert remote.headers['cache-control']=='no-store'
    fresh=PostgresAccountStore(e.dsn)
    from ptb_api.preferences import list_preferences
    assert {item['module']:item['values'] for item in list_preferences(fresh,e.owner)}==SAMPLES

def test_logout_and_expiry_clear_inputs_and_results_but_keep_settings(web):
    e=web;a,_,lease,project=device(e)
    assert patch(a,'M04',SAMPLES['M04']).status_code==200
    uploaded=asset(e,project);generated=asset(e,project)
    e.conn.execute("UPDATE ptb_storage.assets SET kind='result' WHERE id=%s",(generated['id'],))
    assert a.post('/api/v1/auth/logout').status_code==204
    e.workspaces.cleanup()
    assert e.conn.execute('SELECT count(*) AS n FROM ptb_storage.assets WHERE id=ANY(%s::uuid[])',([uploaded['id'],generated['id']],)).fetchone()['n']==0
    b,_,other,_=device(e)
    assert b.get('/api/v1/preferences').json()['modules'][0]['values']==SAMPLES['M04']
    expire(e,other)
    # Settings access neither renews nor requires a content workspace.
    assert b.get('/api/v1/projects').status_code==410
    assert patch(b,'M04',{'draft':{'freq_max_hz':'9000'}},1).status_code==200
    e.workspaces.cleanup()
    saved=b.get('/api/v1/preferences').json()['modules'][0]
    assert saved['revision']==2 and saved['values']['draft']=={'order':'72','amp_max_db':'40','freq_max_hz':'9000'}

def test_account_isolation_csrf_bounds_and_content_rejection(web):
    e=web;a,_,_,_=device(e)
    with TestClient(e.app,base_url=ORIGIN) as anonymous:assert anonymous.get('/api/v1/preferences').status_code==401
    assert a.patch('/api/v1/preferences/M04',json={'expected_revision':0,'changes':{'draft':{'order':'72'}}},headers={'X-CSRF-Token':'wrong'}).status_code==403
    assert a.get('/api/v1/preferences',headers={'X-PTB-Account':'another-account'}).status_code==409
    assert patch(a,'M99',{}).status_code==422
    for values in ({'asset_id':'private-file'},{'draft':{'order':'72','samples':[1,2]}},{'draft':{'order':'data:private'}}):
        result=patch(a,'M04',values);assert result.status_code==422 and 'private-file' not in result.text
    result=a.patch('/api/v1/preferences/M04',content=json.dumps({'padding':'x'*140000}),headers={'Content-Type':'application/json'})
    assert result.status_code==413
    bob=e.accounts.create_user('settings-bob','synthetic-password-only')
    assert e.conn.execute('SELECT count(*) AS n FROM ptb_accounts.module_preferences WHERE owner_id=%s',(bob['id'],)).fetchone()['n']==0

def test_revision_fence_and_nested_merge_do_not_erase_other_devices_choices(web):
    e=web;a,_,_,_=device(e)
    def write(_):
        from ptb_api.preferences import patch_preferences, PreferenceConflict
        try:return patch_preferences(e.accounts,e.owner,'M04',0,{'draft':{'order':'72'}})
        except PreferenceConflict:return None
    with ThreadPoolExecutor(max_workers=2) as pool:assert sum(item is not None for item in pool.map(write,range(2)))==1
    assert patch(a,'M04',{'draft':{'amp_max_db':'45'}},0).status_code==409
    assert patch(a,'M04',{'draft':{'amp_max_db':'45'}},1).status_code==200
    assert a.get('/api/v1/preferences').json()['modules'][0]['values']['draft']=={'order':'72','amp_max_db':'45'}

def test_real_http_independent_chrome_edge_profiles(web):
    from pathlib import Path
    import socket
    import subprocess
    import threading
    import time
    import uvicorn
    from starlette.staticfiles import StaticFiles
    from ptb_api.account_store import PASSWORDS
    from ptb_api.preferences import patch_preferences, list_preferences
    from web_postgres_support import artifacts
    e=web;root=Path(__file__).resolve().parents[2]
    e.conn.execute('UPDATE ptb_accounts.users SET password_hash=%s WHERE id=%s',(PASSWORDS.hash('synthetic-password-only'),e.owner))
    e.accounts.create_user('settings-other','synthetic-password-only')
    for module,values in SAMPLES.items():patch_preferences(e.accounts,e.owner,module,0,values)
    with artifacts.verification_run('web-settings','server-browser','Real new PostgreSQL, same account across owned Chrome/Edge profiles, logout/local-cache-clear/conflict/retry') as (out,scratch):
        with (out/'build.log').open('w',encoding='utf-8') as log:
            subprocess.run(['node',str(root/'frontend/node_modules/vite/bin/vite.js'),'build','--outDir',str(scratch/'frontend')],cwd=root/'frontend',stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
        sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        origin=f'http://127.0.0.1:{port}'
        app=create_app(account_store=e.accounts,auth_settings=AuthSettings(origin=origin,signing_key='isolated-preferences-'*3,allow_insecure_loopback=True),job_store=e.jobs,storage=e.storage,workspace_service=e.workspaces)
        app.mount('/server',StaticFiles(directory=scratch/'frontend',html=True))
        server=uvicorn.Server(uvicorn.Config(app,access_log=False,log_level='warning'))
        thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);thread.start()
        try:
            deadline=time.monotonic()+10
            while not server.started and time.monotonic()<deadline:time.sleep(.02)
            assert server.started
            with (out/'browser.log').open('w',encoding='utf-8') as log:
                subprocess.run(['node',str(root/'tests/e2e/server-preferences.cjs'),origin,str(out),str(scratch)],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            e.workspaces.cleanup()
            saved={row['module']:row['values'] for row in list_preferences(e.accounts,e.owner)}
            assert len(saved)==19 and str(saved['M04']['draft']['order'])=='86'
        finally:
            server.should_exit=True;thread.join(15);sock.close();assert not thread.is_alive()
