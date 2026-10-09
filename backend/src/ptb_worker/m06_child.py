"""Fixed M06 child. Reads one reserved request, emits an atomic bounded bundle."""
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys

MAX_BYTES=24_000_000


def spectrograms(audio,rate):
    """V2 scipy defaults, Tukey .25, 75% overlap, spectrum scaling, percentiles.

    Display raster is bounded separately from numeric synthesis and export.
    """
    import base64
    import numpy as np
    from scipy.signal import spectrogram
    result={}
    for ms in (5,10,20,40):
        n=max(32,min(round(rate*ms/1000),len(audio)));overlap=min(round(n*.75),n-1)
        f,t,sxx=spectrogram(audio,fs=rate,nperseg=n,noverlap=overlap,scaling='spectrum')
        keep=f<=5500;f=f[keep];db=10*np.log10(sxx[keep]+1e-12)
        lo,hi=np.percentile(db,[5,99]);hi=hi if hi>lo else lo+1
        rows=np.linspace(0,len(f)-1,min(256,len(f))).astype(int)
        cols=np.linspace(0,len(t)-1,min(1000,len(t))).astype(int)
        pixels=np.round(255*(1-np.clip((db[np.ix_(rows,cols)]-lo)/(hi-lo),0,1))).astype(np.uint8)
        result[str(ms)]=dict(width=len(cols),height=len(rows),pixels=base64.b64encode(pixels.tobytes()).decode(),
                             t0=float(t[0]),t1=float(t[-1]),fmax=float(f[-1]),low_db=float(lo),high_db=float(hi))
    return result


def compute(header,raw,reaper=None):
    import numpy as np
    from scipy.io import wavfile
    from phonetic_core.models.audio import AudioInput
    from phonetic_core.synthesis.klatt.api import generate,synthesize_with_info,extract,export_parameters,import_parameters
    c=import_parameters(header['parameters']);action=header['action'];seed=header['seed']
    if c['duration']>10 or c['duration']*c['sample_rate']>480000:raise ValueError('m06_admission_budget')
    if hashlib.sha256(raw).hexdigest()!=header['input_sha256']:raise ValueError('m06_input_changed')
    np.random.seed(seed) # Isolated process only; legacy MT19937 random calls preserved.
    files=[];spectra={};diagnostics={}
    if action=='generate':c=generate(c)
    elif action=='extract':
        try:rate,samples=wavfile.read(io.BytesIO(raw))
        except (ValueError,EOFError):raise ValueError('m06_audio_decode_failed') from None
        if samples.ndim>2 or len(samples)>480000 or len(samples)/rate>10 or (samples.ndim==2 and samples.shape[1]>8):raise ValueError('m06_input_budget')
        source=AudioInput(samples,int(rate))
        c=extract(c,source,reaper=reaper,diagnostics=diagnostics)
        if reaper is not None:diagnostics['reaper_binary_sha256']=reaper.sha256
        channels=source.normalized_channels().astype(np.float32);mono=np.mean(channels,axis=1) if channels.ndim>1 else channels
        if action=='extract':spectra=spectrograms(mono.astype(float),int(rate))
    elif action=='synthesize':
        audio,diagnostics=synthesize_with_info(c)
        # Use V2's actual libsndfile conversion. A float64 floor approximation
        # differs by one PCM unit for samples very close to a quantization boundary.
        import soundfile as sf
        stream=io.BytesIO();sf.write(stream,audio.astype(np.float32),c['sample_rate'],format='WAV')
        files.append(('synthesis.wav',stream.getvalue()))
        spectra=spectrograms(audio,c['sample_rate'])
    else:raise ValueError('m06_invalid_action')
    metadata=dict(schema_version='m06/1',action=action,config=c,seed=seed,
                  computation_revision=diagnostics.get('computation_revision','klatt/2'),diagnostics=diagnostics,
                  input_sha256=header['input_sha256'],sample_rate_hz=c['sample_rate'],
                  sample_count=round(c['duration']*c['sample_rate']),
                  curve_time_axis='linspace(0,duration,N)',audio_time_axis='arange(N)/fs',
                  source_ids=['SRC-TDKLATT','REF-KLATT','SRC-PRAAT'],spectrograms=spectra)
    if action=='extract':metadata['curve_time_axis']='arange(N)*0.01; final value held to duration'
    if action=='extract' and c['f0_method']=='reaper':metadata['source_ids'].append('SRC-REAPER')
    files.extend([('m06.ptb.json',json.dumps(metadata,ensure_ascii=False,allow_nan=False).encode()),
                  ('parameters.csv',export_parameters(c).encode('utf8'))])
    return files


def run():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    with open(sys.argv[2],'wb',buffering=0) as stream:
        try:
            path=Path(sys.argv[1])
            if path.stat().st_size>16_000_000:raise ValueError('m06_input_budget')
            line,raw=path.read_bytes().split(b'\n',1);header=json.loads(line)
            if header.get('native_scratch'):
                from .managed_scratch import ReservedNativeScratch
                from .native.reaper import Reaper
                from .io.limits import Limits
                with ReservedNativeScratch(header['native_scratch'],400_000) as scratch:
                    native=Reaper(header['reaper_binary'],scratch,Limits(input_bytes=400_000,output_bytes=2_000_000,process_bytes=1_000_000_000,timeout_seconds=30))
                    files=compute(header,raw,native)
            else:files=compute(header,raw)
            payload=b''.join(v for _,v in files)
            meta=dict(kind='prepared_m06',input_sha256=header['input_sha256'],files=[dict(name=n,size=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in files])
            encoded=json.dumps(meta).encode()
            if len(payload)+len(encoded)+4>MAX_BYTES:raise ValueError('m06_output_budget')
        except Exception as exc:
            code=str(exc).split(':')[0]
            encoded=json.dumps(dict(error=code if code.startswith('m06_') and len(code)<80 else 'm06_execution_failed')).encode();payload=b''
        stream.write(struct.pack('<I',len(encoded)));stream.write(encoded)
        for offset in range(0,len(payload),65536):stream.write(payload[offset:offset+65536])


if __name__=='__main__':run()
