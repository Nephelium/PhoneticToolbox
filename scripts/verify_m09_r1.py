"""M09-R1 owned synthetic workspace, live task bridge and artifact readback."""
import base64
import hashlib
import io
import json
import sqlite3
import sys
import time
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np
import soundfile as sf
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.local_service import LocalService
from ptb_desktop.task_bridge import TaskBridge
from ptb_worker.local_acoustic_files import initialize_local_files

ROOT=Path(__file__).resolve().parents[1]
CORNERS=[dict(x=10/120,y=5/100),dict(x=110/120,y=25/100),dict(x=90/120,y=90/100),dict(x=20/120,y=80/100)]


def setup(label='native'):
    out=ROOT/'output/validation/m09-r1'/(label+'-'+uuid4().hex);out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    gray=np.full((101,121),255,np.uint8)
    cv2.line(gray,(20,30),(100,45),20,2);cv2.line(gray,(25,55),(95,70),70,3)
    for i,(x,y) in enumerate([(10,5),(110,25),(90,90),(20,80)]):cv2.circle(gray,(x,y),4,30+i*50,-1)
    (inputs/'skewed.png').write_bytes(cv2.imencode('.png',gray)[1].tobytes())
    sr=16000;t=np.arange(16001)/sr;audio=np.column_stack([.2*np.sin(2*np.pi*400*t),.12*np.sin(2*np.pi*1700*t)])
    sf.write(inputs/'stereo.wav',audio,sr,subtype='FLOAT')
    return out,db,cache


def wait(service,job):
    until=time.monotonic()+90
    while time.monotonic()<until:
        item=service.get('/api/v1/jobs/'+job['id'])
        if item['state'] not in ('queued','running','cancel_requested'):return item
        time.sleep(.12)
    raise AssertionError('M09 job timeout')


def run():
    out,db,cache=setup();provider=FileProvider();directory=provider.choose('input',lambda:str(out/'inputs'));destination=provider.choose('output',lambda:str(out/'saved'))
    originals={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'inputs').iterdir()}
    report=dict(success=False,checks=[],schema_applied=[],output=str(out));print(out,flush=True)
    try:
        with LocalService(db,local_files_root=cache) as service:
            bridge=TaskBridge(provider,service);files={f['name']:f for f in provider.list(directory['id'])}
            def task(file,config):
                request=dict(op='reconstruct',id=files[file]['id'],config=config,key=uuid4().hex)
                initial=bridge.invoke(request);assert bridge.invoke(request)['id']==initial['id']
                job=wait(service,initial);assert job['state']=='succeeded',job
                data={f['name']:base64.b64decode(bridge.invoke(dict(op='result',job=job['id'],id=f['id']))['base64']) for f in job['result_manifest']['files']}
                for f in job['result_manifest']['files']:assert hashlib.sha256(data[f['name']]).hexdigest()==f['sha256']
                target=out/job['id'];target.mkdir()
                for name,raw in data.items():(target/name).write_bytes(raw)
                result=bridge.invoke(dict(op='save_job',id=job['id'],directory=destination['id']));assert result['count']==4
                return job,data
            image=dict(corners=CORNERS,n_iter=2,time_end=.2)
            job,data=task('skewed.png',image);meta=json.loads(data['reconstruction.ptb.json'])
            assert meta['config']['corners']==CORNERS and meta['geometry']=='perspective-four-corners'
            corrected=cv2.imdecode(np.frombuffer(data['calibrated.png'],np.uint8),0)
            assert corrected.shape==(75,101),corrected.shape
            assert [int(corrected[0,0]),int(corrected[0,-1]),int(corrected[-1,-1]),int(corrected[-1,0])]==[30,80,130,180]
            report['checks'].append('live four corners preserved through Qt task adapter / HTTP / immutable job / bounded child / exported PNG')
            view=bridge.invoke(dict(op='reconstruction_preview',id=files['skewed.png']['id'],config={**image,'mode':'image_draw'}))
            np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(base64.b64decode(view['image_base64']),np.uint8),0),corrected)
            drawing=[dict(color=0,size=10,opacity=.5,points=[dict(x=.25,y=.5),dict(x=.75,y=.5)])]
            _,data=task('skewed.png',{**image,'mode':'image_draw','strokes':drawing})
            assert json.loads(data['reconstruction.ptb.json'])['phase_method']=='griffin-lim'
            report['checks'].append('image preview equals task geometry; edited picture runs Griffin-Lim with source snapshot and nonoverwriting save')
            view=bridge.invoke(dict(op='reconstruction_preview',id=files['stereo.wav']['id'],config={'mode':'audio_draw'}))
            assert view['channels']==2 and view['samples']==16001
            _,data=task('stereo.wav',dict(mode='audio_draw'))
            old,sr=sf.read(out/'inputs/stereo.wav');new,rate=sf.read(io.BytesIO(data['reconstructed.wav']))
            np.testing.assert_allclose(new,old,atol=1e-12,rtol=0);assert sr==rate
            report['checks'].append('real audio preview; unpainted stereo FLOAT export is sample-identical within 1e-12 and exact duration')
            _,data=task('stereo.wav',dict(mode='audio_draw',strokes=drawing))
            new,rate=sf.read(io.BytesIO(data['reconstructed.wav']));np.testing.assert_array_equal(new[:,1],old[:,1]);assert not np.array_equal(new[:,0],old[:,0])
            meta=json.loads(data['reconstruction.ptb.json']);assert meta['n_iter']==0 and meta['phase_method']=='original-stft-phase' and meta['stroke_count']==1
            report['checks'].append('painted original-phase audio changes selected channel, preserves other channel, four artifacts verified and saved')
            large=[dict(color=0,size=2,opacity=.3,points=[dict(x=i/2047,y=.25+.03*np.sin(i*.09+k)) for i in range(2048)]) for k in range(4)]
            job,data=task('stereo.wav',dict(mode='audio_draw',strokes=large))
            meta=json.loads(data['reconstruction.ptb.json']);assert meta['config']['strokes']==large
            with sqlite3.connect(db) as conn:stored=conn.execute('SELECT snapshot FROM jobs WHERE id=?',(job['id'],)).fetchone()[0]
            assert len(stored)<16384 and 'spectral_drawing' in json.loads(stored)['input_refs']
            report['checks'].append('8192-point drawing uses hashed managed input with unchanged DB schema; real generated metadata retains every point exactly')
        assert originals=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'inputs').iterdir()}
        report['success']=True
    finally:
        provider.close();(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))


def serve():
    out,db,cache=setup('browser');provider=FileProvider();directory=provider.choose('input',lambda:str(out/'inputs'));destination=provider.choose('output',lambda:str(out/'saved'))
    with LocalService(db,local_files_root=cache) as service:
        bridge=TaskBridge(provider,service);print(json.dumps(dict(ready=True,out=str(out))),flush=True)
        try:
            for line in sys.stdin:
                request=json.loads(line);body=request['body']
                try:
                    if request['channel']=='task':value=bridge.invoke(body)
                    else:
                        op=body['op']
                        if op=='hello':value=dict(kind='desktop',session=provider.session,api_version='1.1.0',tasks=True)
                        elif op=='fonts':value=['SimSun','Times New Roman','Doulos SIL']
                        elif op=='choose':value=destination if body['purpose']=='output' else directory
                        elif op=='list':value=provider.list(body.get('id') or directory['id'])
                        elif op=='read':
                            raw,sha=provider.read(body['id']);value=dict(base64=base64.b64encode(raw).decode(),sha256=sha)
                        else:raise ValueError('Unknown test transport operation')
                    result=dict(id=request['id'],ok=True,value=value)
                except Exception as exc:result=dict(id=request['id'],ok=False,error=str(exc))
                print(json.dumps(result,ensure_ascii=False),flush=True)
        finally:provider.close()


if __name__=='__main__':serve() if '--serve' in sys.argv else run()
