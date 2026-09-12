"""Real bounded M03 child, four long-file modes, limit rejection and cancellation."""
import hashlib,json,time,os,io
from dataclasses import replace
from pathlib import Path
from uuid import uuid4
from ptb_worker.egg_runtime import command
from ptb_worker.native.windows import InputPipe
from ptb_worker.native.reaper import collect_pipe
from ptb_worker.segmentation import SEGMENT_LIMITS,unpack_bundle
from ptb_worker.io.limits import Cancelled
ROOT=Path(__file__).resolve().parents[1]
os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
def main():
 import numpy as np
 from scipy.io import wavfile
 import struct
 out=ROOT/'output/validation/m03-long'/('process-'+uuid4().hex);out.mkdir(parents=True)
 i=np.arange(120*48000,dtype=np.int64);gain=np.where(i<len(i)//2,1,2)
 samples=np.column_stack([((i%320)*2-320)*70*gain,((i%240)*2-240)*60*gain]).astype(np.int16)
 stream=io.BytesIO();wavfile.write(stream,48000,samples);raw=stream.getvalue()
 assert hashlib.sha256(raw).hexdigest()=='f86756fcefd284e4dbeb524e8d948628119c8b441e6922f4a8d6aeae78f67ee9'
 del samples,i,gain,stream
 limits=replace(SEGMENT_LIMITS,timeout_seconds=240,process_bytes=3_000_000_000)
 report=dict(success=False,checks=[],schema_applied=[],process_bytes=limits.process_bytes,timeout_seconds=limits.timeout_seconds)
 def run(label,config,data=raw,stop=lambda:False):
  request=out/(label+'.request');request.write_bytes(json.dumps(dict(sha256=hashlib.sha256(data).hexdigest(),config=config)).encode()+b'\n'+data)
  pipe=InputPipe();argv=command(request,pipe.name);start=time.monotonic();payload,pid=collect_pipe(argv,pipe,out,limits,stop)
  bundle=unpack_bundle(payload,64_000_000);blobs={f['name']:b for f,b in zip(bundle.manifest['files'],bundle.payloads)}
  target=out/label;target.mkdir()
  for name,b in blobs.items():(target/name).write_bytes(b)
  meta=json.loads(blobs['egg.ptb.json']);assert meta['sample_count']==5_760_000;assert meta['input_sha256']==hashlib.sha256(data).hexdigest()
  report['checks'].append(dict(mode=label,elapsed=time.monotonic()-start,pid=pid,output_bytes=len(payload)));print(label,round(time.monotonic()-start,2),flush=True)
  return meta,blobs
 try:
  meta,blobs=run('preview',dict(mode='preview',roi_start=119.5,roi_end=120,micro_center=119.75));assert meta['preview']['micro_center']==119.75;fs,a=wavfile.read(io.BytesIO(blobs['egg_AUDIO.wav']));assert len(a)==5_760_000 and fs==48000
  meta,blobs=run('single',dict(mode='single'));assert meta['selection']['end_s']==120;assert len(blobs)==5
  for name,b in blobs.items():
   if name.endswith('.png'):assert b[:8]==bytes.fromhex('89504e470d0a1a0a') and all(struct.unpack('>II',b[16:24]))
  assert b'119.' in blobs['egg_DATA.csv']
  meta,blobs=run('batch',dict(mode='batch',generate_images=True));assert len(blobs)==5 and b'119.' in blobs['egg_DATA.csv']
  meta,blobs=run('inverse',dict(mode='inverse',roi_start=119.5,roi_end=119.62));fs,a=wavfile.read(io.BytesIO(blobs['egg_IF.wav']));assert len(a)==5760
  start=time.monotonic()
  try:run('cancel',dict(mode='single'),stop=lambda:time.monotonic()-start>2)
  except Cancelled:report['checks'].append('owned long-file child cancelled and handles closed')
  else:raise AssertionError('Expected cancellation')
  for label,n,fs in [('seconds',121*8000,8000),('frames',5_760_001,96000)]:
   stream=io.BytesIO();wavfile.write(stream,fs,np.zeros((n,2),dtype=np.int16))
   try:run(label,dict(mode='batch'),stream.getvalue())
   except Exception as exc:assert getattr(exc,'code',None)=='egg_input_budget',repr(exc)
   else:raise AssertionError('Expected bounded-input rejection')
   report['checks'].append(label+' budget rejected')
  report['success']=True
 finally:
  (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(out,flush=True)
if __name__=='__main__':main()
