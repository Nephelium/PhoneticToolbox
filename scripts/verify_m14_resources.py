"""M14 fixed-child resource qualification; synthetic input, existing P11 collector."""
import hashlib,json,sys,time
from pathlib import Path
from uuid import uuid4
from ptb_worker.m14_jobs import execute
from ptb_worker.m14_executor import LIMITS,unpack
from ptb_worker.native.windows import InputPipe
from ptb_worker.native.reaper import collect_pipe
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'output/validation/m14/resources'/uuid4().hex;out.mkdir(parents=True)
records=[]
for n in (100,1000,10000):
 raw=('word,ipa,note\n'+''.join(f'字,ma{55 if i%2 else 35},n{i}\n' for i in range(n))).encode()
 preview=json.loads(execute(raw,'public.csv',dict(action='preview',skip_first_row=True,consonant_only_as_zero_initial=True))['m14-preview.json'])
 config=dict(action='export',skip_first_row=True,consonant_only_as_zero_initial=True,settings=preview['config'],font=dict(schema_version='font/1',zh='Microsoft YaHei',latin='Segoe UI',ipa='Doulos SIL',size_px=14))
 header=dict(config=config,name='public.csv',sha256=hashlib.sha256(raw).hexdigest());p=out/f'{n}.input';p.write_bytes(json.dumps(header).encode()+b'\n'+raw);pipe=InputPipe();evidence={};t=time.perf_counter()
 try:
  bundle,_=collect_pipe([sys.executable,'-B','-m','ptb_worker.m14_child',str(p),pipe.name],pipe,out,LIMITS,lambda:False,None,None,evidence)
  values=unpack(bundle,header['sha256'],'export');status='success';sizes={name:len(v) for name,v in values}
 except Exception as e:status=type(e).__name__+':'+str(e);sizes={}
 records.append(dict(rows=n,seconds=time.perf_counter()-t,status=status,sizes=sizes,evidence=evidence));print(records[-1],flush=True)
(out/'report.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8');print(out)
