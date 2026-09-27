"""Per-field comparison, never relaxes the Windows V2 exact reference."""
import json,sys,time,resource,io
import soundfile as sf
from pathlib import Path
import numpy as np,scipy,parselmouth
from scipy.io import wavfile
from phonetic_core.synthesis.klatt.api import defaults,synthesize,extract
from phonetic_core.synthesis.klatt.engine import Engine
from phonetic_core.models.audio import AudioInput
root=Path(__file__).resolve().parents[2];fix=root/'tests/fixtures/m06';meta=json.loads((fix/'v2.json').read_text('utf8'));expected=np.load(fix/'v2.npz');fields={}
def compare(name,actual,reference):
 a=np.asarray(actual);b=np.asarray(reference);finite=np.isfinite(a)&np.isfinite(b);delta=a[finite]-b[finite]
 fields[name]=dict(shape=list(a.shape),exact=bool(np.array_equal(a,b,equal_nan=True)),nan_mask_exact=bool(np.array_equal(np.isnan(a),np.isnan(b))),nonfinite_exact=bool(np.array_equal(a[~finite],b[~finite],equal_nan=True)),mismatches=int(np.count_nonzero((a!=b)&~(np.isnan(a)&np.isnan(b)))),max_abs=float(np.max(np.abs(delta),initial=0)),rms=float(np.sqrt(np.mean(delta*delta))) if delta.size else 0)
for i,case in enumerate(meta['cases']):
 c=defaults()
 for key in ('duration','sequence','sample_rate','curves','silence','boundaries'):c[key]=case[key]
 for name,curve in Engine(c).params.items():compare(str(i)+'_'+name,curve.get_array(c['duration'],c['sample_rate']),expected[str(i)+'_'+name])
 np.random.seed(case['seed']);audio=synthesize(c);compare(str(i)+'_audio',audio,expected[str(i)+'_audio'])
 def pcm(a):
  stream=io.BytesIO();sf.write(stream,a.astype(np.float32),c['sample_rate'],format='WAV');stream.seek(0);return sf.read(stream,dtype='int16')[0]
 compare(str(i)+'_pcm16',pcm(audio),pcm(expected[str(i)+'_audio']))
rate,audio=wavfile.read(fix/'source.wav');c=extract(defaults(),AudioInput(audio,rate))
for name,curve in c['curves'].items():
 compare('extracted_'+name,curve['points'],expected['extracted_'+name]);compare('extracted_'+name+'_time',np.asarray(curve['points'])[:,0],expected['extracted_'+name][:,0]);compare('extracted_'+name+'_value',np.asarray(curve['points'])[:,1],expected['extracted_'+name][:,1])
# Low-level real Linux resource boundary, fixed owned trivial executable only.
from ptb_worker.native.posix import run_bounded
from ptb_worker.io.limits import Limits
boundary={};gate=None
try:run_bounded([sys.executable,'-c','print("m06-boundary")'],b'',root,Limits(timeout_seconds=5),evidence=boundary)
except Exception as e:gate=str(e)
report=dict(platform=sys.platform,versions=dict(numpy=np.__version__,scipy=scipy.__version__,parselmouth=parselmouth.__version__),fields=fields,exact_fields=sum(v['exact'] for v in fields.values()),total_fields=len(fields),task_gate_error=gate,task_gate_evidence=boundary,linux_capability=False)
(root/'output/validation/m06/linux-fields.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='fields'}));print(json.dumps({k:v['max_abs'] for k,v in fields.items() if not v['exact']}))
