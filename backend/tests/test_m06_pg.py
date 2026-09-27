"""M06 real owner/quota/TTL publication on a fresh synthetic PG cluster only."""
import json,time,threading
from uuid import uuid4
import pytest
from test_p07_policy_postgres import cluster,legacy,migrate
from ptb_worker.store import PostgresJobStore,JobError
from ptb_worker.acoustic_files import AcousticFiles
from ptb_worker.m06_task import submit
from ptb_worker.m06_executor import execute_claim
from ptb_api.storage_models import UploadInput
from ptb_api.quota import StorageError
from phonetic_core.synthesis.klatt.api import defaults,export_parameters

def prepare(e):
 migrate(e);store=PostgresJobStore(e.dsn);files=AcousticFiles(store,e.storage)
 c=defaults();c['duration']=.2
 for curve in c['curves'].values():curve['points'][-1][0]=.2
 raw=export_parameters(c).encode();a=e.storage.create(e.owner,UploadInput(project_id=e.project,name='parameters.csv',expected_bytes=len(raw),idempotency_key=uuid4().hex));e.storage.append(e.owner,a['id'],0,raw);a=e.storage.finalize(e.owner,a['id'])
 body=dict(project_id=e.project,idempotency_key=uuid4().hex,action='synthesize',parameters=dict(asset_id=a['id'],sha256=a['sha256']))
 return store,files,body,a

def test_m06_owner_project_actual_quota_and_independent_expiry(legacy):
 e=legacy;store,files,body,a=prepare(e);old=time.time()+3600
 e.conn.execute('UPDATE ptb_storage.assets SET expires_at=%s WHERE id=%s',(old,a['id']))
 other=str(uuid4());project=str(uuid4())
 e.conn.execute("INSERT INTO ptb_accounts.users(id,username,password_hash) VALUES(%s,'m06-other','unused')",(other,));e.conn.execute("INSERT INTO ptb_accounts.projects(id,owner_id,name) VALUES(%s,%s,'other')",(project,other))
 with pytest.raises(JobError):submit(store,other,body)
 with pytest.raises(JobError):submit(store,e.owner,body|{'project_id':project})
 job=submit(store,e.owner,body);claim=store.claim('m06-pg');before=time.time();execute_claim(store,claim,'m06-pg',threading.Event());after=time.time();r=store.get(e.owner,job['id']);assert r['state']=='succeeded',r
 for f in r['result_manifest']['files']:
  assert before+259200<=f['expires_at']<=after+259200
  with pytest.raises(StorageError):e.storage.read_block(other,f['id'],0,10)
  e.storage.read_block(e.owner,f['id'],0,10);assert e.storage.metadata(e.owner,f['id'])['expires_at']==f['expires_at']
 assert r['result_manifest']['policy_version']==2
 usage=e.storage.usage(e.owner);assert usage['used_bytes']==a['size_bytes']+sum(f['size_bytes'] for f in r['result_manifest']['files']);assert usage['reserved_bytes']==0
 assert e.storage.metadata(e.owner,a['id'])['expires_at']==old

def test_m06_full_quota_and_late_generation(legacy):
 e=legacy;store,files,body,a=prepare(e)
 e.conn.execute('UPDATE ptb_storage.quota_accounts SET reserved_bytes=%s-used_bytes WHERE owner_id=%s',(1000000000,e.owner))
 job=submit(store,e.owner,body);claim=store.claim('m06-full');execute_claim(store,claim,'m06-full',threading.Event());r=store.get(e.owner,job['id']);assert r['state']=='failed' and r['error_code']=='quota_exceeded',r
 e.conn.execute('UPDATE ptb_storage.quota_accounts SET reserved_bytes=0 WHERE owner_id=%s',(e.owner,))
 job=submit(store,e.owner,body|{'idempotency_key':uuid4().hex});claim=store.claim('m06-late');store.cancel(e.owner,job['id']);execute_claim(store,claim,'m06-late',threading.Event());assert store.get(e.owner,job['id'])['state']=='cancelled'
 with pytest.raises(StorageError):files.output((claim['id'],'m06-late',claim['generation']),'late.wav','result',1)
