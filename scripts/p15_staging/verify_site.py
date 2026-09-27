"""P15 HTTPS acceptance driver. Default: unauthenticated read-only checks.

--execute-synthetic explicitly performs test-account login/project/upload/delete.
It must only be used after target deployment/test authorization. Credentials are
one JSON line on stdin, never argv or output. Uses no real recordings or files.
No DNS, restart, migrations, system network fault injection or remote protocol.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import time
from urllib.parse import urlsplit
from uuid import uuid4
import wave

import httpx


class CheckFailed(Exception):
    pass


def require(ok,code):
    if not ok: raise CheckFailed(code)


def origin_valid(origin):
    u=urlsplit(origin)
    require(u.scheme=='https' and u.hostname and not any((u.path,u.query,u.fragment,u.username,u.password)),
            'https_origin_required')
    return origin


def synthetic_wav():
    buf=io.BytesIO()
    with wave.open(buf,'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
        f.writeframes(b''.join(struct.pack('<h',round(8000*math.sin(2*math.pi*200*i/16000))) for i in range(1600)))
    return buf.getvalue()


class Checks:
    def __init__(self): self.samples=[]; self.passed=[]; self.owned={}
    def request(self,client,label,method,path,**kwargs):
        started=time.perf_counter()
        r=client.request(method,path,**kwargs)
        self.samples.append({'case':label,'status':r.status_code,
            'client_total_seconds':time.perf_counter()-started,
            'server_timing':r.headers.get('server-timing'),
            'public_rtt_seconds':None})
        # Never serialize request/response headers, cookies, URLs or bodies.
        return r
    def ok(self,label): self.passed.append(label)


def login(c,credential,checks,label):
    challenge=checks.request(c,label+'_challenge','GET','/api/v1/auth/challenge')
    require(challenge.status_code==200,'challenge_failed')
    r=checks.request(c,label+'_login','POST','/api/v1/auth/login',
        json={'username':credential['username'],'password':credential['password']},
        headers={'Origin':str(c.base_url).rstrip('/'),'X-CSRF-Token':challenge.json()['csrf_token']})
    require(r.status_code==200,'login_failed')
    cookies=r.headers.get_list('set-cookie')
    session=[s for s in cookies if s.startswith('__Host-ptb-session=')]
    require(len(session)==1,'session_cookie_name')
    cookie=session[0].lower()
    require(all(v in cookie for v in ('secure','httponly','samesite=strict','path=/')) and 'domain=' not in cookie,
            'session_cookie_flags')
    require(r.headers.get('cache-control')=='no-store','private_response_cache')
    v=r.json()
    return {'Origin':str(c.base_url).rstrip('/'),'X-CSRF-Token':v['csrf_token'],'X-PTB-Account':v['user']['id']}


def public_checks(c,checks,expected_operations):
    health=checks.request(c,'health','GET','/api/v1/health')
    require(health.status_code==200 and health.json().get('mode')=='server','health_mode')
    cap=checks.request(c,'capabilities','GET','/api/v1/capabilities')
    require(cap.status_code==200,'capabilities_failed')
    ops=set(cap.json()['task_operations'])
    require(not ops.intersection({'archive_zip','extract_zip','pitch_manipulation','phonology_induction','spectrogram_to_audio'}),
            'unverified_operation_advertised')
    require(set(expected_operations).issubset(ops),'approved_operation_missing')
    require(checks.request(c,'anonymous_private','GET','/api/v1/projects').status_code==401,'anonymous_access')
    require(checks.request(c,'forged_host','GET','/api/v1/health',headers={'Host':'invalid.example'}).status_code in (400,403,421),
            'forged_host_accepted')
    checks.ok('same_origin_api_and_closed_capabilities')


def lpc_check(a,b,ha,hb,project,asset,digest,checks):
    font={'schema_version':'font/1','zh':'Noto Sans SC','latin':'DejaVu Sans','ipa':'Doulos SIL','size_px':12}
    r=checks.request(a,'lpc_font_preflight','POST','/api/v1/jobs/lpc/fonts',json=font,headers=ha,timeout=335)
    require(r.status_code==200 and r.json()['available'],'lpc_fonts_unavailable')
    body={'schema_version':'m04/1','project_id':project,'idempotency_key':uuid4().hex,
          'audio':{'asset_id':asset,'sha256':digest},'config':{'roi_end':.05,'font':font}}
    r=checks.request(a,'lpc_submit','POST','/api/v1/jobs/lpc/create',json=body,headers=ha)
    require(r.status_code==201,'lpc_submit_failed');job=r.json()['id']
    checks.owned['synthetic_job_id']=job
    r=checks.request(a,'lpc_submit_idempotent','POST','/api/v1/jobs/lpc/create',json=body,headers=ha)
    require(r.status_code==201 and r.json()['id']==job,'lpc_submit_duplicated')
    route='/api/v1/jobs/'+job
    require(checks.request(b,'job_isolation','GET',route,headers=hb).status_code==404,'job_isolation_failed')
    deadline=time.monotonic()+400
    while True:
        r=checks.request(a,'lpc_poll','GET',route,headers=ha)
        require(r.status_code==200,'lpc_poll_failed');state=r.json()
        if state['state'] in ('succeeded','failed','cancelled','interrupted'):break
        if time.monotonic()>deadline:
            checks.request(a,'cancel_owned_timeout','POST',route+'/cancel',headers=ha)
            raise CheckFailed('lpc_poll_deadline')
        time.sleep(.5)
    require(state['state']=='succeeded','lpc_not_succeeded')
    manifest=state['result_manifest']
    require(manifest['complete'] and manifest['policy_version']==2 and len(manifest['files'])==3,'lpc_manifest_invalid')
    require(len({f['id'] for f in manifest['files']})==3 and
            {f['name'] for f in manifest['files']}=={'lpc.ptb.json','lpc_SPECTRUM.png','lpc_AUDIO.wav'},
            'lpc_manifest_file_set')
    checks.owned['synthetic_result_ids']=[f['id'] for f in manifest['files']]
    for f in manifest['files']:
        path='/api/v1/assets/'+f['id']+'/content'
        require(checks.request(b,'result_isolation','GET',path,headers=hb).status_code==404,'result_isolation_failed')
        r=checks.request(a,'lpc_result_hash','GET',path,headers=ha)
        require(r.status_code==200 and len(r.content)==f['size_bytes'] and hashlib.sha256(r.content).hexdigest()==f['sha256'],
                'lpc_result_integrity')
    r=checks.request(a,'result_history','GET','/api/v1/jobs?project_id='+project,headers=ha)
    require(r.status_code==200 and job in [j['id'] for j in r.json()['jobs']],'history_missing')
    checks.ok('lpc_fonts_idempotency_science_three_hashes_history')


def authenticated_checks(a,b,credentials,checks,*,science=False):
    ha=login(a,credentials[0],checks,'a');hb=login(b,credentials[1],checks,'b')
    require(ha['X-PTB-Account']!=hb['X-PTB-Account'],'two_distinct_accounts_required')
    for c,h,label in ((a,ha,'a'),(b,hb,'b')):
        r=checks.request(c,label+'_usage','GET','/api/v1/storage/usage',headers=h)
        require(r.status_code==200,'usage_failed');u=r.json()
        require(u['policy_version']==2 and u['quota_bytes']==1_000_000_000 and u['retention_seconds']==259200,
                'policy_not_active')
        require(not u['over_quota'] and u['ready'] and not u['frozen'],'test_account_not_writable')
    body={'name':'P15 合成验收 '+uuid4().hex[:10]}
    r=checks.request(a,'csrf_missing','POST','/api/v1/projects',json=body,headers={'Origin':ha['Origin']})
    require(r.status_code==403,'csrf_not_enforced')
    r=checks.request(a,'origin_invalid','POST','/api/v1/projects',json=body,headers=ha|{'Origin':'https://invalid.example'})
    require(r.status_code==403,'origin_not_enforced')
    r=checks.request(a,'project_create','POST','/api/v1/projects',json=body,headers=ha)
    require(r.status_code==201,'project_create_failed');project=r.json()['id']
    checks.owned['synthetic_project_id']=project
    r=checks.request(b,'project_isolation','GET','/api/v1/projects',headers=hb)
    require(r.status_code==200 and project not in [p['id'] for p in r.json()['projects']],'project_isolation_failed')
    payload=synthetic_wav();digest=hashlib.sha256(payload).hexdigest()
    request={'project_id':project,'name':'P15 合成 ɑ̃˥.wav','expected_bytes':len(payload),'idempotency_key':uuid4().hex}
    r=checks.request(a,'upload_create','POST','/api/v1/uploads',json=request,headers=ha)
    require(r.status_code==201,'upload_create_failed');asset=r.json()['id']
    checks.owned['synthetic_asset_id']=asset
    route='/api/v1/uploads/'+asset
    r=checks.request(a,'upload_block','PUT',route+'/blocks?offset=0',content=payload,headers=ha|{'Content-Type':'application/octet-stream'})
    require(r.status_code==200,'upload_block_failed')
    r=checks.request(a,'upload_finalize','POST',route+'/finalize',json={'sha256':digest},headers=ha)
    require(r.status_code==200,'upload_finalize_failed');meta=r.json()
    require(meta['policy_version']==2 and meta['sha256']==digest,'uploaded_metadata_failed')
    # HTTP Date is rounded to seconds. Exact clock-boundary verification is a
    # separate isolated DB test; do not pretend this is exact or natural expiry.
    date=r.headers.get('date')
    require(date is not None,'server_date_missing')
    server_time=parsedate_to_datetime(date).timestamp()
    require(abs(meta['expires_at']-server_time-259200)<=2,'ttl_http_window_failed')
    path='/api/v1/assets/'+asset
    for suffix in ('','/content'):
        r=checks.request(b,'asset_isolation','GET',path+suffix,headers=hb)
        require(r.status_code==404,'asset_isolation_failed')
    r=checks.request(a,'download','GET',path+'/content',headers=ha)
    require(r.status_code==200 and r.content==payload,'download_hash_failed')
    r=checks.request(a,'range','GET',path+'/content',headers=ha|{'Range':'bytes=10-29'})
    require(r.status_code==206 and r.content==payload[10:30] and r.headers.get('content-range')==f'bytes 10-29/{len(payload)}',
            'range_failed')
    r=checks.request(a,'range_unsatisfiable','GET',path+'/content',headers=ha|{'Range':f'bytes={len(payload)+1}-'})
    require(r.status_code==416,'range_boundary_failed')
    r=checks.request(a,'download_no_renew','GET',path,headers=ha)
    require(r.status_code==200 and r.json()['expires_at']==meta['expires_at'],'download_renewed_ttl')
    if science:lpc_check(a,b,ha,hb,project,asset,digest,checks)
    # Deletion applies ONLY to this run's ID. No bulk cleanup or project deletion.
    r=checks.request(a,'delete_owned_synthetic','DELETE',path,headers=ha)
    require(r.status_code==200 and r.json()['state']=='deleted','owned_delete_failed')
    for c,h,label in ((a,ha,'a'),(b,hb,'b')):
        require(checks.request(c,label+'_logout','POST','/api/v1/auth/logout',headers=h).status_code==204,'logout_failed')
        require(checks.request(c,label+'_revoked','GET','/api/v1/auth/me').status_code==401,'logout_not_revoked')
    checks.ok('two_accounts_projects_upload_download_range_policy_logout')
    return {'synthetic_project_id':project,'synthetic_asset_id':asset,'input_sha256':digest,'input_bytes':len(payload)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origin',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute-synthetic',action='store_true')
    p.add_argument('--lpc',action='store_true',help='Also submit a fixed public synthetic M04 task; requires --execute-synthetic')
    p.add_argument('--expected-operation',action='append',default=[])
    a=p.parse_args();origin=origin_valid(a.origin)
    require(not a.lpc or a.execute_synthetic,'lpc_requires_synthetic_execution')
    require(not a.output.exists(),'output_must_be_new')
    checks=Checks();result={'schema':'p15-site/1','platform':'real_https_client',
        'origin':origin,'success':False,'executed_synthetic':a.execute_synthetic,
        'at':datetime.now(timezone.utc).isoformat(),
        'not_run':['browser_fonts','legacy_and_overquota','exact_expiry_and_delete_failure',
                   'science_and_service_restart','remote_failover','integrated_load','laboratory_linux']}
    try:
        with httpx.Client(base_url=origin,verify=True,trust_env=False,follow_redirects=False,timeout=35) as c:
            public_checks(c,checks,a.expected_operation)
        if a.execute_synthetic:
            private=json.loads(sys.stdin.readline(16385))
            require(len(private['accounts'])==2,'two_accounts_required')
            require(all(item['username'].startswith('p15_') for item in private['accounts']),'dedicated_p15_accounts_required')
            with httpx.Client(base_url=origin,verify=True,trust_env=False,follow_redirects=False,timeout=35) as x, \
                 httpx.Client(base_url=origin,verify=True,trust_env=False,follow_redirects=False,timeout=35) as y:
                result.update(authenticated_checks(x,y,private['accounts'],checks,science=a.lpc))
        result['success']=True
    except CheckFailed as e: result['failure_code']=str(e)
    except Exception: result['failure_code']='transport_or_contract_error'
    result.update(passed=checks.passed,samples=checks.samples,owned=checks.owned)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'success':result['success'],'report':str(a.output)}))
    return 0 if result['success'] else 2


if __name__=='__main__': sys.exit(main())
