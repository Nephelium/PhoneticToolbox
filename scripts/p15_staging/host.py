"""P15 host adapter. Starting a service is an authorized deployment action.

No DDL, installation or node protocol. Config checks and inventory are read-only.
Open/drain startup invokes Storage.recover, which CAN delete expired assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import sys
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4


class GateError(ValueError):
    pass


KEYS = {'schema','origin','bind','port','unix_user','release_root','frontend',
        'storage_root','private_config','runtime_profile','validation_receipt',
        'release_manifest','control_file','min_free_bytes','max_running',
        'lease_seconds','resource_profile','allowed_operations','remote_enabled'}
OPERATIONS = {'acoustic_analysis','egg_analysis','lpc_analysis'}


def validate_config(c):
    if not isinstance(c,dict) or set(c) != KEYS or c['schema'] != 'p15-staging/1':
        raise GateError('config_schema_invalid')
    u=urlsplit(c['origin'])
    if (u.scheme!='https' or not u.hostname or u.username or u.password or
        u.path or u.query or u.fragment or not re.fullmatch(r'[a-z0-9.-]+',u.hostname) or
        'REPLACE' in c['origin'] or u.port not in (None,443)):
        raise GateError('https_origin_required')
    if (c['bind']!='127.0.0.1' or type(c['port']) is not int or not 1024<=c['port']<=65535 or
        c['max_running']!=1 or c['lease_seconds']!=10 or c['resource_profile']!='server-small' or
        c['remote_enabled'] is not False or not re.fullmatch(r'[a-z][a-z0-9-]{1,31}',c['unix_user'])):
        raise GateError('host_boundary_invalid')
    if (type(c['min_free_bytes']) is not int or c['min_free_bytes']<10*1024**3 or
        not isinstance(c['allowed_operations'],list) or
        any(op not in OPERATIONS for op in c['allowed_operations']) or
        len(set(c['allowed_operations']))!=len(c['allowed_operations'])):
        raise GateError('capability_or_disk_budget_invalid')
    paths={}
    for k in ('release_root','frontend','storage_root','private_config','runtime_profile',
              'validation_receipt','release_manifest','control_file'):
        p=PurePosixPath(c[k])
        if not p.is_absolute() or '..' in p.parts or 'REPLACE' in str(p):
            raise GateError('absolute_target_required')
        paths[k]=p
    if (not paths['frontend'].is_relative_to(paths['release_root']) or
        paths['storage_root'].is_relative_to(paths['release_root']) or
        paths['release_root'].is_relative_to(paths['storage_root']) or
        any(paths[k].is_relative_to(paths['frontend']) for k in
            ('private_config','control_file','runtime_profile','validation_receipt','release_manifest')) or
        paths['control_file'].is_relative_to(paths['storage_root'])):
        raise GateError('private_public_path_overlap')
    return c


def check_manifest(root, manifest, *, required):
    root=Path(root).resolve()
    files=manifest.get('files',{})
    if manifest.get('schema')!='p15-release/1' or not required or not required.issubset(files):
        raise GateError('release_manifest_incomplete')
    for name, digest in files.items():
        p=PurePosixPath(name)
        target=root.joinpath(*p.parts)
        if p.is_absolute() or '..' in p.parts or not target.resolve().is_relative_to(root):
            raise GateError('release_path_invalid')
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise GateError('release_hash_mismatch')


def release_files(root):
    root=Path(root)
    result=set()
    for folder in ('backend/src','packages/phonetic_core/src','frontend/dist','scripts/p15_staging',
                   'backend/migrations','contracts'):
        for path in (root/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                result.add(path.relative_to(root).as_posix())
    return result


def control_mode(path):
    try:
        p=Path(path)
        if p.stat().st_size>1024: return 'blocked'
        c=json.loads(p.read_text('utf-8'))
        return c['mode'] if set(c)=={'mode'} and c['mode'] in ('open','drain','maintenance') else 'blocked'
    except (OSError,ValueError,TypeError,KeyError):
        return 'blocked'


class SiteBoundary:
    """Deployment maintenance and global Host/TLS guard; business auth stays intact."""
    def __init__(self,app,authority,control,allowed_operations=None):
        self.app,self.authority,self.control=app,authority,control
        self.allowed=set(allowed_operations or ())

    async def __call__(self,scope,receive,send):
        if scope['type']!='http': return await self.app(scope,receive,send)
        from starlette.responses import JSONResponse
        async def reject(code,status):
            await JSONResponse({'detail':code},status_code=status,
                headers={'Cache-Control':'no-store'})(scope,receive,send)
        if [v.decode('latin1') for k,v in scope['headers'] if k==b'host'] != [self.authority]:
            return await reject('host_rejected',403)
        if scope['scheme']!='https': return await reject('https_required',400)
        mode=control_mode(self.control)
        path, method=scope['path'],scope['method']
        if mode=='blocked': return await reject('staging_control_unavailable',503)
        auth=path in ('/api/v1/auth/login','/api/v1/auth/logout')
        cancel=path.startswith('/api/v1/jobs/') and path.endswith('/cancel')
        if method not in ('GET','HEAD','OPTIONS') and not auth:
            if mode=='maintenance' or mode=='drain' and not cancel:
                return await reject('staging_draining',503)
        # No remote routes or unreviewed generic/retry/ZIP paths are exposed.
        if path.startswith('/api/v1/worker'):
            return await reject('remote_worker_unavailable',503)
        if method=='POST' and path.startswith('/api/v1/jobs') and not cancel:
            mapping={'/api/v1/jobs/lpc/create':'lpc_analysis','/api/v1/jobs/lpc/fonts':'lpc_analysis',
                     '/api/v1/jobs/egg/create':'egg_analysis','/api/v1/jobs/egg/fonts':'egg_analysis'}
            if path=='/api/v1/jobs/batches/create' and 'acoustic_analysis' in self.allowed:
                chunks=[];size=0
                while True:
                    m=await receive()
                    if m['type']=='http.disconnect': return
                    size+=len(m.get('body',b''))
                    if size>1_000_000: return await reject('request_too_large',413)
                    chunks.append(m.get('body',b''))
                    if not m.get('more_body',False): break
                raw=b''.join(chunks)
                try: operation=json.loads(raw).get('operation')
                except (ValueError,AttributeError): return await reject('invalid_request',422)
                if operation!='acoustic_analysis': return await reject('staging_operation_unavailable',503)
                original_receive=receive; consumed=False
                async def replay():
                    nonlocal consumed
                    if consumed: return await original_receive()
                    consumed=True
                    return {'type':'http.request','body':raw,'more_body':False}
                receive=replay
            elif mapping.get(path) not in self.allowed:
                return await reject('staging_operation_unavailable',503)
        # Reading a stored asset stays possible; computation previews need admission.
        if path.endswith(('/spectrogram','/parameters')) and (not self.allowed or mode!='open'):
            return await reject('staging_operation_unavailable',503)
        started=time.perf_counter()
        async def timed(message):
            if message['type']=='http.response.start':
                message=dict(message)
                message['headers']=list(message.get('headers',[]))+[
                    (b'server-timing',f'app;dur={(time.perf_counter()-started)*1000:.3f}'.encode())]
            await send(message)
        await self.app(scope,receive,timed)


def read_private(path):
    p=Path(path); info=p.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size>16384:
        raise GateError('private_config_invalid')
    if os.name=='posix' and (info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)&0o077):
        raise GateError('private_config_permissions')
    c=json.loads(p.read_text('utf-8'))
    if set(c)!= {'dsn','signing_key','storage_instance_id'}:
        raise GateError('private_config_keys')
    from psycopg.conninfo import conninfo_to_dict
    dsn=conninfo_to_dict(c['dsn'])
    if (dsn.get('host') not in ('127.0.0.1','/var/run/postgresql') or
        dsn.get('hostaddr') not in (None,'127.0.0.1') or
        not dsn.get('dbname') or not dsn.get('user')):
        raise GateError('database_must_be_internal')
    return c


def database_state(dsn,expected_instance):
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(dsn,row_factory=dict_row,connect_timeout=5) as conn:
        conn.execute('SET TRANSACTION READ ONLY')
        conn.execute("SET LOCAL statement_timeout='10s'")
        state=conn.execute('SELECT * FROM ptb_storage.state WHERE singleton').fetchone()
        if not state or str(state['instance_id'])!=expected_instance:
            raise GateError('storage_instance_mismatch')
        if state.get('policy_version',1)!=2:
            raise GateError('storage_policy_migration_required')
        accounts=conn.execute('SELECT count(*) FILTER(WHERE quota_bytes<>1000000000) AS invalid_quota, '
            'count(*) FILTER(WHERE used_bytes+reserved_bytes>quota_bytes) AS over_quota FROM ptb_storage.quota_accounts').fetchone()
        if accounts['invalid_quota']: raise GateError('database_quota_mismatch')
        return {'policy_version':2,'over_quota_accounts':accounts['over_quota'],'frozen':state['frozen']}


def queue_is_supported(jobs,allowed):
    """Conservative deployment hold, not a replacement for B's routing policy.

An unreviewed queued operation holds this server-only worker without claiming it.
Only this deployment's API may submit; B/other claimants must not run alongside.
"""
    with jobs.transaction(write=False) as tx:
        rows=tx.execute("SELECT DISTINCT snapshot::jsonb->>'operation' AS operation FROM {jobs} WHERE state='queued'")
        return all(row['operation'] in allowed for row in rows)


def configure(c):
    if sys.platform!='linux': raise GateError('linux_host_required')
    import pwd
    if pwd.getpwuid(os.getuid()).pw_name!=c['unix_user']: raise GateError('dedicated_uid_required')
    os.environ['PTB_RESOURCE_PROFILE']='server-small'
    os.environ['PTB_LINUX_RUNTIME_PROFILE']=c['runtime_profile']
    os.environ['PTB_LINUX_VALIDATION_RECEIPT']=c['validation_receipt']
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    root=Path(c['release_root'])
    manifest=json.loads(Path(c['release_manifest']).read_text('utf-8'))
    required=release_files(root)|{'backend/src/ptb_api/main.py','backend/src/ptb_worker/executor.py',
        'packages/phonetic_core/src/phonetic_core/__init__.py','frontend/dist/index.html',
        'scripts/p15_staging/host.py','contracts/openapi.json'}
    check_manifest(root,manifest,required=required)
    for package in ('ptb_api','ptb_worker','phonetic_core'):
        import importlib.util
        spec=importlib.util.find_spec(package)
        if not spec or not Path(spec.origin).resolve().is_relative_to(root.resolve()):
            raise GateError('imports_outside_release')


def services(c,private):
    from ptb_api.account_store import PostgresAccountStore
    from ptb_api.storage import Storage
    from ptb_worker.store import PostgresJobStore
    from ptb_worker.acoustic_files import AcousticFiles
    from ptb_worker.acoustic_batches import AcousticBatches
    database_state(private['dsn'],private['storage_instance_id'])
    accounts=PostgresAccountStore(private['dsn']); accounts.check_schema()
    jobs=PostgresJobStore(private['dsn'],max_running=1,lease_seconds=10); jobs.check_schema()
    storage=Storage(private['dsn'],c['storage_root'],min_free_bytes=c['min_free_bytes'])
    files=AcousticFiles(jobs,storage); AcousticBatches(jobs,files)
    if c['allowed_operations']:
        from ptb_worker.native.linux_runtime import load_profile
        _,profile=load_profile(); files.reaper_binary=profile.get('reaper_binary')
        from ptb_worker.native.capabilities import linux_capabilities
        ready,_=linux_capabilities(jobs)
        if set(ready)!=set(c['allowed_operations']): raise GateError('runtime_capability_mismatch')
    mode=control_mode(c['control_file'])
    if mode=='blocked': raise GateError('staging_control_unavailable')
    if mode in ('open','drain'):
        # Explicitly a deployment write/recovery action, never part of inventory.
        storage.recover()
    return accounts,jobs,storage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('role',choices=['api','worker','cleaner','check-config'])
    p.add_argument('--config',type=Path,required=True)
    args=p.parse_args()
    c=validate_config(json.loads(args.config.read_text('utf-8')))
    if args.role=='check-config': print('{"config_valid": true, "deployed": false}'); return
    configure(c); private=read_private(c['private_config'])
    accounts,jobs,storage=services(c,private)
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
    if args.role=='api':
        import uvicorn
        from starlette.staticfiles import StaticFiles
        from ptb_api.main import create_app
        from ptb_api.auth import AuthSettings
        app=create_app('server',account_store=accounts,
            auth_settings=AuthSettings(origin=c['origin'],signing_key=private['signing_key']),
            job_store=jobs,storage=storage)
        app.mount('/server',StaticFiles(directory=c['frontend'],html=True))
        boundary=SiteBoundary(app,urlsplit(c['origin']).netloc,c['control_file'],c['allowed_operations'])
        uvicorn.run(boundary,host='127.0.0.1',port=c['port'],workers=1,access_log=False,
            log_level='critical',proxy_headers=True,forwarded_allow_ips='127.0.0.1',
            limit_concurrency=24,timeout_keep_alive=5,timeout_graceful_shutdown=330)
    elif args.role=='worker':
        from ptb_worker.executor import execute_claim
        worker_id=str(uuid4())
        while not stop.is_set():
            if (not c['allowed_operations'] or control_mode(c['control_file'])!='open' or
                not queue_is_supported(jobs,set(c['allowed_operations']))):
                stop.wait(.5);continue
            claim=jobs.claim(worker_id)
            if claim is None: stop.wait(.1);continue
            # Existing dispatch/fencing remains authoritative. No custom routing.
            operation=json.loads(claim['snapshot'])['operation']
            if operation not in c['allowed_operations']:
                raise GateError('unexpected_queued_operation_stop_for_review')
            execute_claim(jobs,claim,worker_id,stop)
    else:
        while not stop.is_set():
            if control_mode(c['control_file']) in ('open','drain'):
                try: storage.cleanup()
                except Exception: print('storage_cleanup_retry_required',flush=True)
            stop.wait(1)


if __name__=='__main__':
    try: main()
    except Exception:
        # Exceptions/DSNs/HTTP bodies must never escape to service logs.
        print('P15 startup or service gate failed; review private configuration and evidence.',file=sys.stderr)
        raise SystemExit(2) from None
