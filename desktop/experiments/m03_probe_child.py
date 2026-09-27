"""Fixed probe workload; runs only in the independently staged EGG interpreter."""
import ctypes
from ctypes import wintypes
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys

payload=Path(__file__).resolve().parent
prefix=Path(sys.prefix).resolve()
os.environ['PATH']=os.pathsep.join([str(prefix),str(prefix/'Library/bin'),str(Path(os.environ['SystemRoot'])/'System32')])
os.environ['MPLCONFIGDIR']=str(Path(sys.argv[1])/'mpl-cache')
dll_handle=os.add_dll_directory(str(prefix/'Library/bin'))
sys.path.insert(0,str(payload/'backend'))

def dll_paths():
    psapi=ctypes.WinDLL('psapi',use_last_error=True);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE;handle=kernel.GetCurrentProcess()
    modules=(wintypes.HMODULE*4096)();needed=wintypes.DWORD()
    psapi.EnumProcessModules.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD)]
    psapi.GetModuleFileNameExW.argtypes=[wintypes.HANDLE,wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD]
    if not psapi.EnumProcessModules(handle,modules,ctypes.sizeof(modules),ctypes.byref(needed)):raise ctypes.WinError()
    if needed.value>ctypes.sizeof(modules):raise ValueError('Module inventory budget')
    result=[]
    for module in modules[:needed.value//ctypes.sizeof(wintypes.HMODULE)]:
        value=ctypes.create_unicode_buffer(32768)
        if not psapi.GetModuleFileNameExW(handle,module,value,len(value)):raise ctypes.WinError()
        result.append(str(Path(value.value).resolve()))
    return sorted(result)

def main():
    import numpy as np
    from scipy.io import wavfile
    from ptb_worker.egg_runtime import fingerprint
    from ptb_worker.fonts import check_fonts
    from ptb_api.font_models import FigureFontSnapshot
    from ptb_worker.egg_child import prepare
    out=Path(sys.argv[1]);report=dict(success=False,prefix=str(prefix),runtime=fingerprint(),fonts=check_fonts(FigureFontSnapshot()),modes={})
    if not report['fonts']['available']:raise ValueError('Required fonts unavailable')
    with np.load(payload/'fixture.npz',allow_pickle=False) as data:samples=np.column_stack([data['load.egg_signal_raw'],data['load.audio_signal']])
    stream=io.BytesIO();wavfile.write(stream,44100,samples);raw=stream.getvalue()
    for mode in ('preview','single','batch','inverse'):
        config=dict(mode=mode,font=FigureFontSnapshot().model_dump())
        if mode!='batch':config.update(roi_start=.1,roi_end=.22)
        bundle=prepare(raw,config,'探针 ɑ̃˥.wav');size=struct.unpack('<Q',bundle[:8])[0];header=json.loads(bundle[8:8+size]);offset=8+size;files={}
        target=out/mode;target.mkdir()
        for f in header['files']:
            blob=bundle[offset:offset+f['size_bytes']];offset+=f['size_bytes'];digest=hashlib.sha256(blob).hexdigest()
            if digest!=f['sha256']:raise ValueError('Bundle digest mismatch')
            (target/f['name']).write_bytes(blob);files[f['name']]=digest
        if offset!=len(bundle):raise ValueError('Bundle size mismatch')
        report['modes'][mode]=files
    report['modules']=sorted({str(Path(m.__file__).resolve()) for m in sys.modules.values() if getattr(m,'__file__',None) and Path(m.__file__).is_file()})
    report['dlls']=dll_paths();report['sys_path']=sys.path;report['success']=True
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

try:main()
except Exception as exc:
    (Path(sys.argv[1])/'failure.json').write_text(json.dumps({'type':type(exc).__name__,'error':str(exc)},ensure_ascii=False),encoding='utf-8')
    raise
