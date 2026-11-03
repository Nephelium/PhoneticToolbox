"""Complete installed manuals through the frozen authenticated Web application."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import zipfile

import uvicorn
from starlette.staticfiles import StaticFiles
from ptb_api.account_store import PASSWORDS
from ptb_api.auth import AuthSettings
from ptb_api.main import create_app
from test_account_preferences_postgres import web
from test_p07_policy_postgres import legacy
from web_postgres_support import cluster, artifacts
from web_candidate_support import prepare_frontend, verify_frontend_unchanged


ROOT=Path(__file__).resolve().parents[2]
CANDIDATE=ROOT/'output/manual-packs/3.1.2-manual-20261030-2c2f8e7a/candidate-manifest.json'


def expected_candidate(manifest_path:Path,frontend:Path)->dict:
    """Read independent expectations from actual candidate bytes, never the UI store."""
    manifest_bytes=manifest_path.read_bytes()
    manifest=json.loads(manifest_bytes)
    assert manifest['schemaVersion']=='ptb-manual-local-candidates/1'
    assert manifest['published'] is False
    base_bytes=(frontend/'manual/project.json').read_bytes()
    base=json.loads(base_bytes)
    host_assets={asset['id']:asset for asset in base['assets']}
    production_feed=json.loads((frontend/'manual/language-packs.json').read_bytes())
    assert production_feed['published'] is False and production_feed['packs']==[]
    packs=[]
    for candidate in manifest['packs']:
        archive=(manifest_path.parent/candidate['file']).resolve(strict=True)
        assert archive.parent==manifest_path.parent.resolve()
        assert archive.stat().st_size==candidate['bytes']
        with archive.open('rb')as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==candidate['sha256']
        with zipfile.ZipFile(archive)as bundle:
            package=json.loads(bundle.read('language-pack.json'))
            for name,digest in package['hashes'].items():
                assert hashlib.sha256(bundle.read(name)).hexdigest()==digest,name
            project=json.loads(bundle.read('project.json'))
            assert project['sourceDigest']==base['sourceDigest']==candidate['sourceDigest']
            assert project['id']==base['id']==package['projectId']
            assert project['language']==candidate['locale']==package['locale']
            overrides=package['assetOverrides']
            assets={asset['id']:asset for asset in project['assets']}
            assert len(overrides)==343
            assert len(set(assets)-set(overrides))==40
            # The manifest registers Chinese source identities even when a
            # language override shadows them; only unshadowed IDs need media.
            assert set(assets)-set(overrides)<=set(package['sharedAssets'])<=set(assets)
            expected_assets={}
            for identity,asset in assets.items():
                host=host_assets[identity]
                digest=package['hashes'][overrides[identity]] if identity in overrides else host['sha256']
                expected_assets[identity]={'id':identity,'kind':asset['kind'],'sha256':digest,
                    'override':identity in overrides,'hostPath':host['path']}
            chapters=[]
            for descriptor in project['chapters']:
                chapter=json.loads(bundle.read(descriptor['path']))
                media=[]
                def walk(node):
                    if node['type'] in ('image','audio','video'):
                        identity=node.get('attrs',{}).get('assetId')
                        assert identity in expected_assets,(chapter['id'],identity)
                        assert expected_assets[identity]['kind']==node['type']
                        media.append({'id':identity,'kind':node['type']})
                    for child in node.get('content',[]):walk(child)
                walk(chapter['body'])
                assert chapter['id']==descriptor['id'] and chapter['title']==descriptor['title']
                chapters.append({'id':chapter['id'],'title':chapter['title'],'media':media})
            assert len(chapters)==22
            assert {item['id'] for chapter in chapters for item in chapter['media']}==set(assets)
            packs.append({**candidate,'path':str(archive),'chapters':chapters,'assets':expected_assets})
    assert {pack['locale'] for pack in packs}=={'en','ja'}
    return {'manifest':str(manifest_path),'manifestSha256':hashlib.sha256(manifest_bytes).hexdigest(),
        'baseProjectSha256':hashlib.sha256(base_bytes).hexdigest(),'baseAssetCount':len(host_assets),
        'projectId':base['id'],'sourceDigest':base['sourceDigest'],'packs':packs}


def test_complete_manual_local_install_and_offline_reopen_in_frozen_web(web):
    assert os.environ.get('PTB_VERIFY_FRONTEND_DIST'),'Set PTB_VERIFY_FRONTEND_DIST to the frozen Web candidate; this test never builds source.'
    e=web
    e.conn.execute('UPDATE ptb_accounts.users SET password_hash=%s WHERE id=%s',
        (PASSWORDS.hash('synthetic-password-only'),e.owner))
    with artifacts.verification_run('manual-candidate-web','authenticated-offline',
            'Frozen Web candidate, real Chrome login and home/help controls, final EN/JA local ZIPs, all 22 chapters/media with full networking disabled and real help close/reopen; no injected app store or source harness')as(out,scratch):
        frontend,identity=prepare_frontend(ROOT,scratch,out)
        assert identity['explicitCandidate']
        expected=expected_candidate(CANDIDATE,frontend)
        expected['testSources']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('backend/tests/test_manual_candidate_web.py','tests/e2e/manual-candidate-web.cjs')}
        fixture=scratch/'manual-candidate.json'
        fixture.write_text(json.dumps(expected,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (out/'candidate-input.json').write_text(json.dumps({
            key:value for key,value in expected.items()if key!='packs'
        }|{'packs':[{key:pack[key]for key in ('locale','path','sha256','bytes','version','sourceDigest')}for pack in expected['packs']]},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        sock=socket.socket();sock.bind(('127.0.0.1',0))
        origin=f'http://127.0.0.1:{sock.getsockname()[1]}'
        app=create_app(account_store=e.accounts,
            auth_settings=AuthSettings(origin=origin,signing_key='isolated-frozen-manual-'*3,allow_insecure_loopback=True),
            job_store=e.jobs,storage=e.storage,workspace_service=e.workspaces,
            vocal_resources=ROOT/'resources/vocal_tract/native')
        app.mount('/server',StaticFiles(directory=frontend,html=True))
        server=uvicorn.Server(uvicorn.Config(app,access_log=False,log_level='warning'))
        thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);thread.start()
        try:
            deadline=time.monotonic()+15
            while not server.started and time.monotonic()<deadline:time.sleep(.02)
            assert server.started
            with(out/'browser.log').open('w',encoding='utf-8')as log:
                subprocess.run(['node',str(ROOT/'tests/e2e/manual-candidate-web.cjs'),origin,str(out),str(scratch),str(fixture)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            for pack in expected['packs']:
                with Path(pack['path']).open('rb')as stream:
                    assert hashlib.file_digest(stream,'sha256').hexdigest()==pack['sha256']
            assert hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()==expected['manifestSha256']
        finally:
            server.should_exit=True;thread.join(15);sock.close()
            verify_frontend_unchanged(frontend,identity,out)
            assert not thread.is_alive()
