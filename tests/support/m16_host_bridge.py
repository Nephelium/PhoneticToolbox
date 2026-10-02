"""Chrome test transport to production M16 service, with synthetic audio only."""
import json
import sys
from pathlib import Path
from uuid import uuid4
from ptb_desktop.recording.service import RecordingService
from m16_test_backend import Backend

ROOT=Path(__file__).resolve().parents[2]

def main():
    out=ROOT/'output/validation/m16'/('chrome-'+uuid4().hex);out.mkdir(parents=True);project=out/'project';project.mkdir();exports=out/'exports';exports.mkdir()
    service=RecordingService(backend=Backend());print(json.dumps({'ready':True,'out':str(out)}),flush=True)
    try:
        for line in sys.stdin:
            r=json.loads(line);body=r['body']
            try:
                if r['channel']=='recording':value=service.dispatch(body)
                else:
                    op=body['op']
                    if op=='hello':value={'kind':'desktop','session':'m16-synthetic-test','api_version':'1.1.0','tasks':False}
                    elif op=='fonts':value=['Arial','Microsoft YaHei','Doulos SIL']
                    elif op=='m16_choose':value=service.grant(exports if body['purpose']=='export' else project,body['purpose'])
                    else:raise ValueError('Unsupported isolated test transport '+op)
                response={'id':r['id'],'ok':True,'value':value}
            except Exception as exc:response={'id':r['id'],'ok':False,'error':str(exc)}
            print(json.dumps(response,ensure_ascii=False,allow_nan=False),flush=True)
    finally:service.close()

if __name__=='__main__':main()
