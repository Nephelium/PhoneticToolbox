"""Read back P17 products from an owned host run, without executing computation."""
import argparse,hashlib,io,json
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();folder=a.run/'workspace/files';state=json.loads((folder/'.ptb-local.json').read_text('utf8'));groups={};checks=[]
 for f in state['assets'].values():
  if f['state']!='ready' or f['kind']!='result':continue
  raw=(folder/(f['id']+'.bin')).read_bytes();assert len(raw)==f['size_bytes'];assert hashlib.sha256(raw).hexdigest()==f['sha256'];groups.setdefault(f['job_id'],{})[f['name']]=raw
  if f['name'].endswith('.wav'):
   samples,rate=sf.read(io.BytesIO(raw));info=sf.info(io.BytesIO(raw));assert np.isfinite(samples).all();checks.append(dict(job=f['job_id'],file=f['name'],sample_rate=rate,frames=len(samples),channels=info.channels,subtype=info.subtype,sha256=f['sha256']))
 m07=0;m07_backends=[]
 for job,files in groups.items():
  if 'm07.ptb.json' in files:
   meta=json.loads(files['m07.ptb.json']);m07_backends.append(meta['analysis']['f0_backend'])
   if meta['action']=='generate':
    parts=[]
    for name in sorted(n for n in files if n.startswith('step') and n.endswith('.wav')):
     v,sr=sf.read(io.BytesIO(files[name]),dtype='int16');parts.append(v)
    v,sr=sf.read(io.BytesIO(files['combined_steps.wav']),dtype='int16');np.testing.assert_array_equal(v,np.concatenate(parts));assert len(parts)==meta['generation']['step_count'];m07+=1
  if 'reconstruction.ptb.json' in files:
   meta=json.loads(files['reconstruction.ptb.json']);v,sr=sf.read(io.BytesIO(files['reconstructed.wav']));assert sr==meta['sample_rate'] and len(v)==meta['samples'];assert len(files)==4
 report=dict(success=True,result_groups=len(groups),wav_count=len(checks),m07_backends=sorted(set(m07_backends)),m07_groups_with_exact_concatenation=m07,wavs=checks)
 (a.run/'readback-report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in report.items()if k!='wavs'}))
if __name__=='__main__':main()
