"""Measure the actual Windows bounded worker at short and admitted edge inputs."""
import json,time,io,threading
from uuid import uuid4
from scipy.io import wavfile
import numpy as np
from verify_m06_wiring import setup,ROOT
from ptb_worker.store import SQLiteJobStore,LOCAL_PROJECT
from ptb_worker.local_acoustic_files import LocalAcousticFiles
from ptb_worker.m06_task import submit
from ptb_worker.m06_executor import execute_claim
from phonetic_core.synthesis.klatt.api import defaults,export_parameters
out,db,cache=setup();store=SQLiteJobStore(db);files=LocalAcousticFiles(store,cache);rows=[]
for action,duration,rate in [('synthesize',.1,16000),('synthesize',10,16000),('synthesize',10,48000),('extract',10,48000),('synthesize',10.01,16000)]:
 c=defaults();c['duration']=duration;c['sample_rate']=rate
 for curve in c['curves'].values():curve['points'][-1][0]=duration
 parameters=files.import_input(export_parameters(c).encode(),'m06-parameters.csv','table')
 body=dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,action=action,parameters=parameters)
 if action=='extract':
  stream=io.BytesIO();t=np.arange(round(duration*rate))/rate;wavfile.write(stream,rate,(.2*np.sin(2*np.pi*150*t)+.08*np.sin(2*np.pi*300*t)).astype(np.float32));body['audio']=files.import_input(stream.getvalue(),'boundary.wav','audio')
 job=submit(store,'local',body);claim=store.claim('m06-resource');assert claim['id']==job['id'];evidence={};start=time.monotonic()
 execute_claim(store,claim,'m06-resource',threading.Event(),evidence=evidence);r=store.get('local',job['id'])
 row=dict(action=action,duration=duration,sample_rate=rate,elapsed_seconds=time.monotonic()-start,state=r['state'],error=r['error_code'],evidence=evidence)
 if r['result_manifest']:row['output_bytes']=sum(f['size_bytes'] for f in r['result_manifest']['files'])
 rows.append(row);print(json.dumps(row),flush=True)
 assert r['state']==('failed' if duration>10 else 'succeeded'),r
(out/'resources.json').write_text(json.dumps(rows,indent=2),encoding='utf8');print(out)
