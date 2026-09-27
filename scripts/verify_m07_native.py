"""Independent original/native REAPER and migrated adapter comparison."""
from pathlib import Path
import sys,json,warnings,subprocess
import numpy as np
from scipy.io import wavfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests/support'))
import m07_baseline as old
from phonetic_core.manipulation import m07_api as core
from phonetic_core.manipulation.m07_models import PhonationAnalysisConfig,PhonationGenerationConfig
from ptb_worker.m07_science import estimator
from ptb_worker.native.reaper import Reaper
from ptb_worker.io.scratch import Scratch
out=ROOT/'output/validation/m07/native';out.mkdir(exist_ok=True)
# Hide this original capture's owned native executable without changing its inputs.
original_run=subprocess.run
def hidden_run(*args,**kwargs):kwargs['creationflags']=subprocess.CREATE_NO_WINDOW;return original_run(*args,**kwargs)
subprocess.run=hidden_run
warnings.filterwarnings('error',message='.*[Ff]allback.*')
pairs=[];svc=old.PhonationSynthesisService()
with Scratch(out,400000) as scratch:
 native=Reaper(ROOT/'phonetic_toolbox/core/acoustic/reaper.exe',scratch)
 for i,hz in enumerate((120,180)):
  t=np.arange(16000)/16000;a=sum(np.sin(2*np.pi*hz*k*t)/k for k in range(1,30))*.15;pcm=np.round(a*32767).astype(np.int16);path=out/f'input{i}.wav';wavfile.write(path,16000,pcm)
  original=svc.analyze_file(path,old.PhonationAnalysisConfig(f0_backend='reaper'));migrated=core.analyze(pcm.astype(float)/32768,16000,PhonationAnalysisConfig(f0_backend='reaper'),estimator(native))
  for key in ('signal','f0_hz','lpc_coefficients','residual','pulses'):np.testing.assert_array_equal(getattr(original,key),getattr(migrated,key))
  pairs.append((original,migrated))
 for reverse in (False,True):
  pair=pairs[::-1] if reverse else pairs
  for kind in (1,2,3):
   a=svc.generate_continuum(pair[0][0],pair[1][0],old.ContinuumType(kind),old.PhonationGenerationConfig(step_count=3));b=core.generate(pairs[0][1],pairs[1][1],kind,PhonationGenerationConfig(step_count=3),reverse);np.testing.assert_array_equal(a.audio_steps,b.audio_steps)
(out/'report.json').write_text(json.dumps(dict(native_original_and_migrated_exact=True,inputs=2,groups=6,python_fallback=False),indent=2),'utf8');print(out)
