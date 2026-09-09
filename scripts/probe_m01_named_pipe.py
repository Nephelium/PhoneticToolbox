"""M01-C design gate only: exact native executable + small synthetic input.

Source: Microsoft named-pipe documentation; no changes to user/system settings.
"""
import ctypes as c
from ctypes import wintypes as w
import hashlib
import json
from pathlib import Path
import subprocess
import time
from uuid import uuid4
from baseline_support import RECIPES, create_fixture
from scipy.io import wavfile
from phonetic_core.models.audio import AudioInput
from phonetic_core.acoustic.reaper_codec import reaper_pcm16


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/'output/validation/m01'/('pipe-probe-'+uuid4().hex)
    folder.mkdir()
    binary=root/'phonetic_toolbox/core/acoustic/reaper.exe'
    assert hashlib.sha256(binary.read_bytes()).hexdigest()=='279fecc82ed0a49b0277b114270771d7670299068e849058b392672825981824'
    original=create_fixture(folder,RECIPES[0]);fs,data=wavfile.read(original)
    converted=folder/'input.wav';wavfile.write(converted,16000,reaper_pcm16(AudioInput(data,fs)))
    kernel=c.WinDLL('kernel32',use_last_error=True)
    kernel.CreateNamedPipeW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,w.DWORD,w.DWORD,w.DWORD,w.DWORD,c.c_void_p]
    kernel.CreateNamedPipeW.restype=w.HANDLE
    kernel.ConnectNamedPipe.argtypes=[w.HANDLE,c.c_void_p]
    kernel.ReadFile.argtypes=[w.HANDLE,c.c_void_p,w.DWORD,c.POINTER(w.DWORD),c.c_void_p]
    kernel.CloseHandle.argtypes=[w.HANDLE]
    rows=[]
    for limit in [100000,128]:
        name='\\\\.\\pipe\\ptb-m01-'+uuid4().hex
        handle=kernel.CreateNamedPipeW(name,1|0x80000,1|8,1,4096,4096,0,None)
        assert handle!=c.c_void_p(-1).value,c.get_last_error()
        proc=None;output=bytearray();status='pending';started=time.monotonic()
        try:
            proc=subprocess.Popen([str(binary),'-i',str(converted),'-f',name,'-a','-e','.005','-m','60','-x','880','-t'],
                cwd=folder,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            while time.monotonic()-started<8:
                kernel.ConnectNamedPipe(handle,None)
                buf=c.create_string_buffer(min(4096,limit-len(output)+1));n=w.DWORD()
                ok=kernel.ReadFile(handle,buf,len(buf),c.byref(n),None)
                error=c.get_last_error()
                if n.value:
                    if len(output)+n.value>limit: status='budget_exceeded';break
                    output.extend(buf.raw[:n.value])
                elif proc.poll() is not None and error in [109,233,232,536]:
                    status='completed' if proc.returncode==0 else 'native_failed';break
                time.sleep(.005)
            else: status='timeout'
        finally:
            if proc and proc.poll() is None: proc.kill()
            if proc: proc.wait(timeout=3)
            kernel.CloseHandle(handle)
        rows.append({'limit':limit,'status':status,'bytes_received':len(output),'exit_code':proc.returncode,
            'header':bytes(output[:100]).decode('ascii',errors='replace'),'seconds':time.monotonic()-started})
        (folder/f'output-{limit}.f0').write_bytes(output)
    (folder/'report.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps({'folder':folder.relative_to(root).as_posix(),'cases':rows}))


if __name__=='__main__':main()
