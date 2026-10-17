"""Explicitly authorized isolated Linux test host; never edits production services.

Fixed dedicated user, fresh database, loopback listeners and separate feedback DB.
Run prepare once, existing verify_web_linux science/flow, then start. No cleanup.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import pwd
import secrets
import subprocess
import sys

ROOT = Path('/srv/ptb-staging/releases/web-integration-20261016')
STATE = Path('/var/lib/ptb-staging')
USER = 'ptb-staging'
PYTHON = ROOT / 'venv/bin/python'


def owned(args, **kwargs):
    uid = pwd.getpwnam(USER).pw_uid
    env = ['HOME='+str(STATE), f'XDG_RUNTIME_DIR=/run/user/{uid}',
           f'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/{uid}/bus',
           f'PYTHONPATH={ROOT}/backend/src:{ROOT}/packages/phonetic_core/src',
           'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1']
    return subprocess.run(['sudo','-u',USER,'env',*env,*map(str,args)],check=True,**kwargs)


def save(path, value, owner=USER):
    info=pwd.getpwnam(owner)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
    path.chmod(0o600);os.chown(path,info.pw_uid,info.pw_gid)


def baseline():
    files=[Path('/var/www/phonetictoolbox-coming-soon/index.html')]
    files += [p for p in Path('/etc/nginx').rglob('*') if p.is_file()]
    return dict(files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},
                services=subprocess.check_output(['systemctl','show','nginx','ptb-feedback','-p','Id,MainPID,ActiveState'],text=True))


def prepare():
    for name in ('pgdata','private.json','manual-web.json','manual-feedback.json'):
        assert not (STATE/name).exists(), 'Existing isolated state must be reviewed, never overwritten'
    assert not any((STATE/'assets').iterdir()), 'Assets must be empty before fresh setup'
    (STATE/'assets').chmod(0o700)
    save(STATE/'reports/manual-production-before.json',baseline())
    owned(['/usr/lib/postgresql/16/bin/initdb','-D',STATE/'pgdata','--auth-local=trust','--auth-host=reject','--no-locale','--encoding=UTF8'],stdout=subprocess.DEVNULL)
    owned(['/usr/lib/postgresql/16/bin/pg_ctl','-D',STATE/'pgdata','-l',STATE/'reports/manual-postgres.log',
           '-o',f"-k {STATE}/pgsocket -p 18767 -c listen_addresses='' -c unix_socket_permissions=0700 -c shared_buffers=128MB -c max_connections=64",'start'],stdout=subprocess.DEVNULL)
    owned(['/usr/lib/postgresql/16/bin/createdb','-h',STATE/'pgsocket','-p','18767','ptb_web_staging'])
    owned([PYTHON,'-B',ROOT/'scripts/verify_web_linux.py','setup'])
    sys.path.insert(0,str(ROOT/'backend/src'))
    from ptb_api.feedback_mail import SMTPMailer,settings_from_script
    production=json.loads(Path('/etc/phonetictoolbox-feedback/private.json').read_text('utf-8-sig'))
    smtp=settings_from_script(production['smtpSource']) if production.get('smtpSource') else production['smtp']
    SMTPMailer(smtp).check()  # Authentication only. No automated real email is sent.
    private=json.loads((STATE/'private.json').read_text())
    internal_key=secrets.token_urlsafe(40);admin_key=secrets.token_urlsafe(40)
    web=dict(private,enable_registration=True,enable_web_workspaces=True,enable_jobs=True,
             enable_file_jobs=True,enable_acoustic_batches=True,enable_worker=True,
             reaper_binary=str(ROOT/'reaper'),smtp=smtp,entry_path='/app/',web_admin_key=internal_key)
    save(STATE/'manual-web.json',web)
    owned(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','7','-subj','/CN=PhoneticToolbox isolated test',
           '-addext','subjectAltName=IP:127.0.0.1','-keyout',STATE/'manual-tls.key','-out',STATE/'manual-tls.pem'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    (STATE/'manual-tls.key').chmod(0o600)
    save(STATE/'manual-feedback.json',dict(database=str(STATE/'manual-feedback.sqlite'),secret=secrets.token_urlsafe(40),
         adminToken=admin_key,port=18768,allowUnverified=True,tls_certfile=str(STATE/'manual-tls.pem'),tls_keyfile=str(STATE/'manual-tls.key'),
         webOperations=dict(url='http://127.0.0.1:18765/api/v1/_ops',origin='https://127.0.0.1:18765',key=internal_key)))
    # A synthetic login for manual testing. The user can also register normally.
    password=secrets.token_urlsafe(16)
    account=dict(username='jingjing_test',password=password)
    seed="""import json,sys
from ptb_api.registration_store import PostgresRegistrationStore
from ptb_api.account_store import PASSWORDS,hash_token
from secrets import token_urlsafe
v=json.loads(sys.stdin.read());s=PostgresRegistrationStore(v['dsn']);t=token_urlsafe(32)
assert s.begin_registration(v['username'],'jingjing-test@example.test',PASSWORDS.hash(v['password']),hash_token(t))
assert s.consume_email(hash_token(t),'register')
"""
    owned([PYTHON,'-B','-c',seed],input=json.dumps(dict(account,dsn=private['dsn'])),text=True)
    export=Path('/home/admin/ptb-isolated-client-20261016');export.mkdir(mode=0o700,exist_ok=False)
    info=pwd.getpwnam('admin');os.chown(export,info.pw_uid,info.pw_gid)
    save(export/'client.json',dict(web_origin='http://127.0.0.1:18765',manager_url='https://127.0.0.1:18768/api/v1/feedback',
         manager_token=admin_key,test_account=account),owner='admin')
    (export/'ca.pem').write_bytes((STATE/'manual-tls.pem').read_bytes());os.chown(export/'ca.pem',info.pw_uid,info.pw_gid)
    save(STATE/'reports/manual-prepared.json',dict(success=True,loopback_only=True,smtp_authentication=True,email_sent=False,
         database='fresh isolated ptb_web_staging',production_unchanged=baseline()==json.loads((STATE/'reports/manual-production-before.json').read_text())))
    print('ISOLATED_PREPARED; no real email sent; no production service changed',flush=True)


def start():
    receipt=json.loads((STATE/'validated-runtime.json').read_text())
    assert receipt['profile_sha256']==hashlib.sha256((STATE/'runtime.json').read_bytes()).hexdigest()
    assert set(receipt['operations'])=={'lpc_analysis','egg_analysis','acoustic_analysis'}
    for role,command in [
        ('web',[PYTHON,'-B','-m','ptb_api.server','--port','18765','--frontend',ROOT/'frontend/dist','--config',STATE/'manual-web.json']),
        ('feedback',[PYTHON,'-B',ROOT/'deployment/feedback/server.py','--config',STATE/'manual-feedback.json'])]:
        owned(['systemd-run','--user','--unit=ptb-isolated-'+role,'--collect',
               '--property=RuntimeMaxSec=86400','--property=MemoryMax='+('768M' if role=='web' else '256M'),
               '--property=CPUQuota='+('100%' if role=='web' else '25%'),
               '--setenv=PYTHONPATH='+str(ROOT/'backend/src')+':'+str(ROOT/'packages/phonetic_core/src'),
               '--setenv=PTB_RESOURCE_PROFILE=server-small','--setenv=PTB_LINUX_RUNTIME_PROFILE='+str(STATE/'runtime.json'),
               '--setenv=PTB_LINUX_VALIDATION_RECEIPT='+str(STATE/'validated-runtime.json'),
               '--setenv=MPLCONFIGDIR='+str(STATE/'matplotlib-cache'),
               '--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',*command])
    print('ISOLATED_STARTED; API and feedback units stop automatically after 24 hours',flush=True)


if __name__=='__main__':
    assert sys.platform=='linux' and os.getuid()==0 and Path(__file__).resolve().parents[1]==ROOT
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','start']);args=parser.parse_args()
    globals()[args.phase]()
