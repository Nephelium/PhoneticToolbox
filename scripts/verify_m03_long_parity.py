"""Independent V2 full-file evidence for M03-E3-B; no source or user-data writes."""
import os,sys,json,hashlib,subprocess,time
from pathlib import Path
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[1]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def worker(kind,wav,out):
 import numpy as np
 from scipy.io import wavfile
 if kind=='v2':
  assert sys.dont_write_bytecode
  sys.path.insert(0,str(ROOT.parent/'PhoneticToolbox_v2'))
  from phonetic_toolbox.models.config import EGGConfig
  from phonetic_toolbox.services.egg_service import EGGAnalysisService
  cfg=EGGConfig();cfg.gci_method='slope';cfg.goi_method='scale';svc=EGGAnalysisService()
  result=svc.analyze_events(svc.load_file(str(wav),cfg),cfg);svc.calculate_praat_f0(result)
  cq=lambda a,b:svc.calculate_cq_sq_segment(result,a,b,cfg)
 else:
  from phonetic_core.egg import EGGConfig,prepare,analyze_events,cq_segment
  from phonetic_core.egg.f0 import praat_pitch
  cfg=EGGConfig.for_workbench();fs,samples=wavfile.read(wav);result=analyze_events(prepare(samples,fs,cfg),cfg)
  track=praat_pitch(result.audio_signal,fs);result.audio_f0_times=track.legacy_times;result.audio_f0_values=track.values
  cq=lambda a,b:cq_segment(result,a,b,cfg)
 arrays={k:getattr(result,k) for k in ['time_vector','egg_signal_raw','egg_signal_processed','audio_signal','gci_times','goi_times','peak_times','gci_f0_times','gci_f0_values','audio_f0_times','audio_f0_values']}
 duration=len(result.time_vector)/result.fs
 for label,a,b in [('full',0,duration),('tail',duration-.5,duration),('head',0,.5)]:
  for k,v in zip(['time','cq','sq'],cq(a,b)):arrays[label+'.'+k]=v
 def digest(v):
  a=np.asarray(v if v is not None else [],dtype=np.float64);return dict(size=a.size,sha256=sha(a.tobytes()))
 report={k:digest(v) for k,v in arrays.items()};out.write_text(json.dumps(report,indent=2),encoding='utf-8')
def main():
 if len(sys.argv)>1 and sys.argv[1]=='--worker':return worker(sys.argv[2],Path(sys.argv[3]),Path(sys.argv[4]))
 import numpy as np
 from scipy.io import wavfile
 out=ROOT/'output/validation/m03-long'/('parity-'+uuid4().hex);out.mkdir(parents=True)
 n=120*48000;i=np.arange(n,dtype=np.int64);phase=i%320;wave=((phase*2-320)*70);gain=np.where(i<n//2,1,2)
 samples=np.column_stack([wave*gain,((i%240)*2-240)*60*gain]).astype(np.int16);wav=out/'long.wav';wavfile.write(wav,48000,samples)
 v2=Path(json.loads((ROOT/'output/validation/p03/context-before.json').read_text('utf-8'))['v2_environment'])
 results=[]
 for label,prefix,kind in [('v2-1',v2,'v2'),('v2-2',v2,'v2'),('core',ROOT/'.venv/m03-compatible','core')]:
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',QT_QPA_PLATFORM='offscreen');env['PATH']=os.pathsep.join([str(prefix),str(prefix/'Library/bin'),str(prefix/'Scripts'),env.get('PATH','')])
  t=time.monotonic()
  with (out/(label+'.log')).open('w',encoding='utf-8') as log:subprocess.run([str(prefix/'python.exe'),'-B','-X','utf8',str(Path(__file__).resolve()),'--worker',kind,str(wav),str(out/(label+'.json'))],env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240,creationflags=subprocess.CREATE_NO_WINDOW)
  results.append(json.loads((out/(label+'.json')).read_text('utf-8')));print(label,round(time.monotonic()-t,2),flush=True)
 assert results[0]==results[1]==results[2]
 report=dict(success=True,arrays=len(results[0]),values=sum(v['size'] for v in results[0].values()),input_sha256=sha(wav.read_bytes()),input_recipe='120s 48k PCM16 periodic sawtooths with doubled amplitude in second half',source_sha256=sha((ROOT.parent/'PhoneticToolbox_v2/phonetic_toolbox/services/egg_service.py').read_bytes()))
 (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(out,flush=True)
if __name__=='__main__':main()
