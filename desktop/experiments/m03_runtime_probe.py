"""Local-only one-file runtime probe; no GUI, downloads, installs or databases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from ptb_worker.native.windows import OwnedProcess

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=180)
    if not getattr(sys,'frozen',False):parser.add_argument('--payload',type=Path,required=True)
    args=parser.parse_args()
    if not .01<=args.timeout<=240:parser.error('timeout must be between .01 and 240 seconds')
    payload=(Path(sys._MEIPASS)/'payload' if getattr(sys,'frozen',False) else args.payload).resolve()
    output=args.output.resolve()
    if output.exists():parser.error('output must be a new directory')
    output.mkdir(parents=True)
    manifest=json.loads((payload/'manifest.json').read_text('utf-8'))
    for item in manifest['files']:
        p=(payload/item['path']).resolve()
        if not p.is_relative_to(payload) or not p.is_file() or p.stat().st_size!=item['size'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Payload verification failed')
    for name in list(os.environ):
        if name.startswith(('PYTHON','PTB_')):os.environ.pop(name)
    os.environ['PATH']=str(Path(os.environ['SystemRoot'])/'System32')
    os.environ['OPENBLAS_NUM_THREADS']=os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='1'
    # Do not let the PyInstaller host DLL directory affect the standard child Python.
    import ctypes
    ctypes.windll.kernel32.SetDllDirectoryW(None)
    argv=[str(payload/'runtime/python.exe'),'-I','-B',str(payload/'probe_child.py'),str(output)]
    start=time.monotonic();process=OwnedProcess(argv,output,3_000_000_000)
    (output/'process.json').write_text(json.dumps(dict(parent=os.getpid(),child=process.pid,payload=str(payload))),encoding='utf-8')
    try:
        while process.poll() is None:
            if time.monotonic()-start>args.timeout:raise TimeoutError('Probe timeout')
            time.sleep(.05)
        if process.poll()!=0:raise RuntimeError('Probe child failed; see failure.json')
    finally:process.close()
    report=json.loads((output/'report.json').read_text('utf-8'))
    allowed=[payload,Path(os.environ['SystemRoot']).resolve()]
    unexpected=[p for p in report['modules']+report['dlls'] if not any(Path(p).is_relative_to(root) for root in allowed)]
    if unexpected:raise ValueError('Child loaded outside runtime/system: '+repr(unexpected))
    if any(not Path(p).is_relative_to(payload) for p in report['sys_path']):raise ValueError('Unexpected Python import root')
    summary=dict(success=True,seconds=time.monotonic()-start,files=len(manifest['files']),payload_bytes=sum(f['size'] for f in manifest['files']),loaded_modules=len(report['modules']),loaded_dlls=len(report['dlls']),local_only=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))

if __name__=='__main__':main()
