"""Real local HTTP preview auth, lifecycle and unchanged output acceptance."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SOURCES=[str(ROOT/p) for p in ('backend/src','packages/phonetic_core/src')]
sys.path[:0]=SOURCES
os.environ['PYTHONPATH']=os.pathsep.join(SOURCES)
from fastapi.testclient import TestClient
from ptb_api.main import create_app
from ptb_worker.spectrogram_session import SpectrogramSession
from ptb_worker.spectrogram_preview import render


def main():
    inventory=json.loads((ROOT/'output/validation/p17/natural-inventory.json').read_text(encoding='utf8'))
    raw=Path(inventory['selected']['medium']['path']).read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    instances=[]
    def create_session():
        instance=SpectrogramSession();instances.append(instance);return instance
    headers={'Authorization':'Bearer p17-local-test','Origin':'http://127.0.0.1:9917','Content-Type':'application/octet-stream'}
    path='/api/v1/preview/spectrogram'
    view=dict(channel=0,start=.1,end=.6,width=800)
    expected=render(raw,**view)
    timings=[];checks=[]
    with patch('ptb_worker.spectrogram_session.SpectrogramSession',create_session):
        with TestClient(create_app(mode='local',local_token='p17-local-test',local_origin=headers['Origin'])) as client:
            assert client.post(path,params=view,content=raw).status_code==403
            assert instances[0].process is None
            checks.append('unauthorized request starts no worker')
            for i in range(24):
                t=time.perf_counter();response=client.post(path,params=view,content=raw,headers=headers)
                timings.append((time.perf_counter()-t)*1000)
                assert response.status_code==200,response.text
                actual=response.json();assert actual.pop('sha256')==sha
                assert actual==expected
                assert response.headers['cache-control']=='no-store'
            process=instances[0].process
            checks.append('24 real HTTP results exactly match one-shot reference')
            invalid=client.post(path,params={**view,'channel':99},content=raw,headers=headers)
            assert invalid.status_code==422
        assert process.poll() is not None and instances[0].process is None
        checks.append('API lifespan closes its owned worker')
        with TestClient(create_app(mode='server')) as server:
            response=server.post(path,params=view,content=raw,headers=headers)
            assert response.status_code in (401,403,404)
        assert len(instances)==1,'Hosted server must not create local session'
        checks.append('server path does not enable local session')
    assert hashlib.sha256(Path(inventory['selected']['medium']['path']).read_bytes()).hexdigest()==sha
    warm=sorted(timings[1:]);report=dict(checks=checks,cold_ms=timings[0],warm_ms=timings[1:],warm_p95=warm[int(.95*(len(warm)-1))],original_unchanged=True)
    (ROOT/'output/validation/p17/spectrogram-http.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
