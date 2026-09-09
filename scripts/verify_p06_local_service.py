"""Use the already approved SQLite state through real owned local HTTP services."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4
import urllib.error
from ptb_desktop.local_service import LocalService

ROOT=Path(__file__).resolve().parents[1]
PROJECT='00000000-0000-4000-8000-000000000001'


def wait_for(check,timeout=15):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        result=check()
        if result:return result
        time.sleep(0.1)
    raise AssertionError('Local task timed out')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-test-data',action='store_true')
    if not parser.parse_args().approved_test_data:parser.error('Requires approved isolated local state')
    path=ROOT/'output/validation/p06/local-state.sqlite3'
    assert path.is_file()
    first=LocalService(jobs_path=path)
    try:
        first.start()
        assert first.get('/api/v1/capabilities')['task_operations']==['pipeline_check']
        body={'project_id':PROJECT,'idempotency_key':uuid4().hex,'operation':'pipeline_check','config':{'sample_count':257,'seed':0}}
        job=first.request('/api/v1/jobs','POST',body)
        assert first.request('/api/v1/jobs','POST',body)['id']==job['id']
        result=wait_for(lambda:r if (r:=first.get('/api/v1/jobs/'+job['id']))['state']=='succeeded' else None)
        assert result['result_manifest']['sha256']==hashlib.sha256(bytes(range(256))+b'\x00').hexdigest()
        events=first.get('/api/v1/jobs/'+job['id']+'/events')['events']
        assert events[-1]['state']=='succeeded'
        old_url=first.url
    finally:first.close()
    assert first.exit_code==0
    second=LocalService(jobs_path=path)
    try:
        second.start()
        assert second.token!=first.token
        assert second.get('/api/v1/jobs/'+job['id'])['result_manifest']==result['result_manifest']
        try:second.get('/api/v1/auth/me')
        except urllib.error.HTTPError as error:assert error.code==404
        else:raise AssertionError('Local accounts must never exist')
    finally:second.close()
    assert second.exit_code==0
    report={'actual_loopback_http':True,'idempotent_submission':True,'core_result_matches_known_bytes':True,
            'events_and_service_restart_restore':True,'local_accounts_unavailable':True,'service_exit_codes':[first.exit_code,second.exit_code],
            'scope':'Local API + SQLite + independent worker/core process; full Qt interaction not claimed'}
    (ROOT/'output/validation/p06/local-service.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
