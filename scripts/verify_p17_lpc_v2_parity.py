"""Compare saved real LPC output with V2 original function under its compatible MKL runtime."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
root=Path(__file__).resolve().parents[1];out=Path(sys.argv[1]).resolve();f=next((out/'saved').glob('LPC*.json'));data=json.loads(f.read_text('utf8'));fs,raw=wavfile.read(out/'inputs'/data['input_name']);mono=raw.astype(np.float64)/32768
p=root.parent/'PhoneticToolbox_v2/phonetic_toolbox/core/acoustic/lpc.py';spec=importlib.util.spec_from_file_location('p17_original_v2_lpc',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);s=data['selection'];freq,mag=m.compute_lpc_spectrum(mono[s['start_sample']:s['end_sample']],fs,data['config']['order']);assert np.array_equal(freq,np.array(data['spectrum']['frequencies_hz']));assert np.array_equal(mag,np.array(data['spectrum']['magnitude_db']));result={'v2_original_function':str(p),'frequency_values':len(freq),'magnitude_values':len(mag),'byte_exact':True};(out/'lpc-v2-parity.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(result)
