"""Resource evidence from actual bounded child tasks, synthetic inputs only."""
import sys,json,time,io,threading
from pathlib import Path
import numpy as np
from scipy.io import wavfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend/tests'))
from test_m07 import runtime,body,run
from uuid import uuid4
root=Path(__file__).resolve().parents[1];out=root/'output/validation/m07/resources'/uuid4().hex;out.mkdir(parents=True)
r=runtime.__wrapped__(out);records=[]
def measure(label,request):
 start=time.monotonic();done=threading.Event();samples={'temporary_peak_bytes':0,'temporary_reserved_peak_bytes':0};started=[]
 def sample():
  while not done.is_set():
   with r[1].locked() as state:
    assets=[a for a in state['assets'].values() if a['kind']=='temporary' and a['state']!='deleted']
    samples['temporary_peak_bytes']=max(samples['temporary_peak_bytes'],sum(r[1]._path(a['id']).stat().st_size for a in assets))
    samples['temporary_reserved_peak_bytes']=max(samples['temporary_reserved_peak_bytes'],sum(a['size_bytes']+a['reserved_bytes'] for a in assets))
   done.wait(.02)
 thread=threading.Thread(target=sample);thread.start()
 try:job,ev=run(r,request,on_started=lambda pid:started.append(time.monotonic()-start))
 finally:done.set();thread.join()
 ev.update(samples,dispatch_seconds_to_child_created=started[0] if started else None);ev.update(label=label,state=job['state'],error=job['error_code'],wall_seconds=time.monotonic()-start,result_bytes=sum(f['size_bytes'] for f in (job.get('result_manifest') or {}).get('files',[])))
 records.append(ev);(out/'report.json').write_text(json.dumps(records,indent=2),'utf8');assert job['state']=='succeeded',job;return job
small=measure('startup+typical analysis',body(r))
for reverse in (False,True):
 for kind in (1,2,3):measure(f'typical 9 steps {reverse}/{kind}',body(r,'generate',analysis_job_id=small['id'],reverse_direction=reverse,continuum_type=kind))
for i,hz in enumerate((120,175)):
 t=np.arange(480000)/48000;audio=.3*np.sin(2*np.pi*hz*t)+.1*np.sin(4*np.pi*hz*t);b=io.BytesIO();wavfile.write(b,48000,np.round(audio*32767).astype(np.int16));r[2][i]=r[1].import_input(b.getvalue(),f'10seconds{i}.wav','audio')
large=measure('10seconds 480000frames analysis',body(r));measure('10seconds 50steps',body(r,'generate',analysis_job_id=large['id'],generation={'step_count':50}))
print(out)
