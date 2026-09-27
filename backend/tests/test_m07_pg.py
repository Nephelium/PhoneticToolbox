"""M07 real science on a fresh synthetic PostgreSQL cluster. No existing DB."""
import hashlib
import io
import json
import threading
import time
from pathlib import Path
from uuid import uuid4
import pytest
from scipy.io import wavfile
from test_p07_policy_postgres import cluster,legacy,migrate
from ptb_worker.store import PostgresJobStore,JobError
from ptb_worker.acoustic_files import AcousticFiles
from ptb_worker.m07_task import submit
from ptb_worker.m07_executor import execute_claim
from ptb_api.storage_models import UploadInput
from ptb_api.quota import StorageError
ROOT=Path(__file__).resolve().parents[2]

def prepare(e):
    migrate(e);store=PostgresJobStore(e.dsn);files=AcousticFiles(store,e.storage);refs=[]
    for i in range(2):
        raw=(ROOT/f'output/validation/m07/baseline/round1/input{i}.wav').read_bytes()
        asset=e.storage.create(e.owner,UploadInput(project_id=e.project,name=f'input{i}.wav',expected_bytes=len(raw),idempotency_key=uuid4().hex))
        e.storage.append(e.owner,asset['id'],0,raw);asset=e.storage.finalize(e.owner,asset['id']);refs.append(dict(asset_id=asset['id'],sha256=asset['sha256']))
    return store,files,dict(project_id=e.project,idempotency_key=uuid4().hex,action='analyze',source=refs[0],target=refs[1])

def run(e,store,body):
    job=submit(store,e.owner,body);claim=store.claim('m07-pg');execute_claim(store,claim,'m07-pg',threading.Event());return store.get(e.owner,job['id'])

def data(e,result):
    values={}
    for f in result['result_manifest']['files']:
        raw=b''.join(e.storage.read_block(e.owner,f['id'],i,min(262144,f['size_bytes']-i)) for i in range(0,f['size_bytes'],262144))
        assert hashlib.sha256(raw).hexdigest()==f['sha256'];values[f['name']]=raw
    return values

def test_m07_two_owners_actual_analysis_generation_and_retention(legacy):
    e=legacy;store,files,body=prepare(e);other=str(uuid4());project=str(uuid4())
    e.conn.execute("INSERT INTO ptb_accounts.users(id,username,password_hash) VALUES(%s,'m07-other','unused')",(other,));e.conn.execute("INSERT INTO ptb_accounts.projects(id,owner_id,name) VALUES(%s,%s,'other')",(project,other))
    with pytest.raises(JobError):submit(store,other,body)
    with pytest.raises(JobError):submit(store,e.owner,body|{'project_id':project})
    original_expiry=time.time()+3600
    e.conn.execute('UPDATE ptb_storage.assets SET expires_at=%s WHERE id IN (%s,%s)',(original_expiry,body['source']['asset_id'],body['target']['asset_id']))
    analysis=run(e,store,body);assert analysis['state']=='succeeded',analysis
    controls=json.loads(data(e,analysis)['m07.ptb.json'])['controls'];controls['source'][5]+=10
    applied=run(e,store,body|dict(idempotency_key=uuid4().hex,action='apply',analysis_job_id=analysis['id'],controls=controls));assert applied['state']=='succeeded',applied
    before=time.time();result=run(e,store,body|dict(idempotency_key=uuid4().hex,action='generate',analysis_job_id=applied['id'],generation={'step_count':3}));after=time.time();assert result['state']=='succeeded',result
    raw=data(e,result);fs,a=wavfile.read(io.BytesIO(raw['combined_steps.wav']));assert fs==11025 and len(a)>0
    assert result['result_manifest']['policy_version']==2
    for f in result['result_manifest']['files']:
        assert before+259200<=f['expires_at']<=after+259200
        with pytest.raises(StorageError):e.storage.read_block(other,f['id'],0,1)
        assert e.storage.metadata(e.owner,f['id'])['expires_at']==f['expires_at']
    assert e.storage.metadata(e.owner,body['source']['asset_id'])['expires_at']==original_expiry
    usage=e.storage.usage(e.owner);assert usage['reserved_bytes']==0
    expected=e.conn.execute("SELECT COALESCE(sum(size_bytes),0) AS n FROM ptb_storage.assets WHERE owner_id=%s AND state='ready'",(e.owner,)).fetchone()['n'];assert usage['used_bytes']==expected

def test_m07_quota_expiry_and_late_fencing(legacy):
    e=legacy;store,files,body=prepare(e)
    e.conn.execute('UPDATE ptb_storage.quota_accounts SET reserved_bytes=%s-used_bytes WHERE owner_id=%s',(1000000000,e.owner))
    failed=run(e,store,body);assert failed['state']=='failed' and failed['error_code']=='quota_exceeded',failed
    e.conn.execute('UPDATE ptb_storage.quota_accounts SET reserved_bytes=0 WHERE owner_id=%s',(e.owner,))
    body['idempotency_key']=uuid4().hex;job=submit(store,e.owner,body);claim=store.claim('m07-old');store.cancel(e.owner,job['id']);execute_claim(store,claim,'m07-old',threading.Event());assert store.get(e.owner,job['id'])['state']=='cancelled'
    with pytest.raises(StorageError):files.output((job['id'],'m07-old',claim['generation']),'late.wav','result',1)
    e.conn.execute('UPDATE ptb_storage.assets SET expires_at=%s WHERE id=%s',(time.time()-1,body['target']['asset_id']))
    with pytest.raises(JobError):submit(store,e.owner,body|{'idempotency_key':uuid4().hex})
