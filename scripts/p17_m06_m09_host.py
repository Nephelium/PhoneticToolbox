"""P17 owned real-input host. No audio generation or existing-schema mutation."""
import argparse,base64,hashlib,json,sys,time
from pathlib import Path
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.task_bridge import TaskBridge
from ptb_desktop.local_service import LocalService
from ptb_worker.local_workspace import prepare_workspace
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--inputs',type=Path,nargs='+',required=True);args=p.parse_args()
 out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir();manifest=[]
 import soundfile as sf
 for i,source in enumerate(args.inputs):
  source=source.resolve();allowed=Path.home()/'Desktop/project/音频数据'
  if not source.is_relative_to(allowed.resolve()):raise ValueError('Input outside authorized recording directory')
  raw=source.read_bytes();dest=inputs/f'真实录音{i+1}.wav';dest.write_bytes(raw);info=sf.info(source)
  manifest.append(dict(relative=str(source.relative_to(allowed)),name=dest.name,sha256=hashlib.sha256(raw).hexdigest(),rate=info.samplerate,channels=info.channels,frames=info.frames,duration=info.duration))
 (out/'inputs.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 # M09 uses a spectrogram calculated from the first authentic recording.
 import numpy as np
 from scipy.signal import spectrogram
 import cv2
 samples,rate=sf.read(args.inputs[0]);samples=samples[:,0] if samples.ndim>1 else samples
 freqs,times,power=spectrogram(samples,rate,nperseg=256,noverlap=192,mode='magnitude');db=20*np.log10(np.maximum(power,1e-15)/max(float(power.max()),1e-15));pixels=np.flipud(np.round(255*(1-np.clip((db+30)/30,0,1))).astype('uint8'));(inputs/'真实录音语谱图.png').write_bytes(cv2.imencode('.png',pixels)[1].tobytes())
 for extension in ['.jpg','.bmp']:(inputs/('真实录音语谱图'+extension)).write_bytes(cv2.imencode(extension,pixels)[1].tobytes())
 empty=out/'empty-inputs';empty.mkdir()
 dbpath,cache=prepare_workspace(out/'workspace',ROOT/'backend/migrations');provider=FileProvider();directory=provider.choose('input',lambda:str(inputs));destination=provider.choose('output',lambda:str(saved));empty_directory=provider.choose('input',lambda:str(empty));timings=[]
 with LocalService(dbpath,local_files_root=cache,reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe') as service:
  bridge=TaskBridge(provider,service);print(json.dumps(dict(ready=True,out=str(out),manifest=manifest,empty_directory=empty_directory)),flush=True)
  try:
   for line in sys.stdin:
    request=json.loads(line);body=request['body'];start=time.perf_counter()
    try:
     if request['channel']=='task':value=bridge.invoke(body)
     else:
      op=body['op']
      if op=='hello':value=dict(kind='desktop',session=provider.session,api_version='1.1.0',tasks=True)
      elif op=='fonts':value=['Arial','Microsoft YaHei','Doulos SIL']
      elif op=='choose':value=destination if body['purpose']=='output' else directory
      elif op=='list':value=provider.list(body.get('id') or directory['id'])
      elif op=='read':
       raw,sha=provider.read(body['id']);value=dict(base64=base64.b64encode(raw).decode(),sha256=sha)
      else:raise ValueError('Unsupported P17 transport operation')
     response=dict(id=request['id'],ok=True,value=value)
    except Exception as exc:response=dict(id=request['id'],ok=False,error=str(exc))
    timings.append(dict(op=body.get('op'),action=body.get('action'),ms=(time.perf_counter()-start)*1000,ok=response['ok']))
    print(json.dumps(response,ensure_ascii=False),flush=True)
  finally:(out/'transport-timings.json').write_text(json.dumps(timings,indent=2),encoding='utf8')
 for item,source in zip(manifest,args.inputs):assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256']
if __name__=='__main__':main()
