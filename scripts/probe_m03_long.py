"""P04/M03-E3-B isolated candidate budget probe; never a product entry."""
import ctypes as c,json,os,sys,time,hashlib,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
prefix=Path(sys.prefix);os.environ['PATH']=os.pathsep.join([str(prefix),str(prefix/'Library/bin'),os.environ.get('PATH','')]);dll=os.add_dll_directory(str(prefix/'Library/bin'))
sys.path.insert(0,str(ROOT/'backend/src'))
import numpy as np
from scipy.io import wavfile
from ptb_worker import egg_child
from ptb_worker.segmentation import unpack_bundle
import matplotlib
if "--chunked" in sys.argv:matplotlib.rcParams["agg.path.chunksize"]=10000
seconds=float(sys.argv[1]);mode=sys.argv[2];out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True)
fs=48000;n=int(seconds*fs);t=np.arange(n)/fs
amp=.2+.6*(t/seconds);egg=np.sin(2*np.pi*150*t)+.4*np.sin(2*np.pi*300*t);audio=np.sin(2*np.pi*150*t)
samples=np.column_stack([egg*amp,audio*amp]);samples=(samples/np.max(np.abs(samples))*30000).astype(np.int16)
wav=io.BytesIO();wavfile.write(wav,fs,samples);raw=wav.getvalue();del samples,t,amp,egg,audio,wav
# Explicit candidate probe only. Published limits remain untouched until measured.
egg_child.MAX_SECONDS=600;egg_child.MAX_SAMPLES=28_800_000
config=dict(mode=mode,keep_praat_f0=True,keep_gci_f0=True)
if mode=='preview':config.update(roi_start=seconds-.5,roi_end=seconds)
start=time.monotonic();payload=egg_child.prepare(raw,config);elapsed=time.monotonic()-start
bundle=unpack_bundle(payload,64_000_000)
class Counters(c.Structure):
 _fields_=[('cb',c.c_ulong),('faults',c.c_ulong)]+[(k,c.c_size_t) for k in ['peak_ws','ws','peak_pp','pp','peak_np','np','pagefile','peak_pagefile','private']]
info=Counters();info.cb=c.sizeof(info);kernel=c.WinDLL('kernel32');kernel.GetCurrentProcess.restype=c.c_void_p
get=c.WinDLL('psapi').GetProcessMemoryInfo;get.argtypes=[c.c_void_p,c.c_void_p,c.c_ulong];assert get(kernel.GetCurrentProcess(),c.byref(info),info.cb)
report=dict(seconds=seconds,mode=mode,elapsed=elapsed,peak_commit=info.peak_pagefile,peak_working_set=info.peak_ws,output_bytes=len(payload),input_sha256=hashlib.sha256(raw).hexdigest(),input_bytes=len(raw),files=bundle.manifest['files'])
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
