"""Test-only stdin RPC to real owned FileProvider/TaskBridge; no new HTTP host."""
import base64,json,os,sqlite3,sys
from pathlib import Path
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.local_service import LocalService
from ptb_desktop.task_bridge import TaskBridge
from ptb_worker.local_acoustic_files import initialize_local_files

ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'output/validation/m03-ui'/('chrome-'+uuid4().hex);out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone();source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache);inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    if '--real-only' in sys.argv:
        private=Path(r'C:\Users\13680\Desktop\project\音频数据\EGG测试\3.wav')
        raw=private.read_bytes()
        (inputs/'3.wav').write_bytes(raw)
        (inputs/'3-复测.wav').write_bytes(raw)
    else:
        with np.load(ROOT/'tests/fixtures/m03/EGG-SYN-PCM16.npz') as a:samples=np.column_stack([a['load.egg_signal_raw'],a['load.audio_signal']])
        wavfile.write(inputs/'EGG ɑ̃˥.wav',44100,samples);wavfile.write(inputs/'mono.wav',44100,samples[:,0]);wavfile.write(inputs/'silence.wav',44100,np.zeros_like(samples))
    if '--realtime-input' in sys.argv:
        # Explicit opt-in private regression input, only inside ignored evidence.
        private=Path(sys.argv[sys.argv.index('--realtime-input')+1])
        (inputs/'realtime.wav').write_bytes(private.read_bytes())
    if '--ranges' in sys.argv:
        wavfile.write(inputs/'wide.wav',44100,np.tile(samples,(8,1)))
        wavfile.write(inputs/'long.wav',44100,np.tile(samples,(83,1)))
    if '--long' in sys.argv:
        i=np.arange(120*48000,dtype=np.int64);gain=np.where(i<len(i)//2,1,2)
        samples=np.column_stack([((i%320)*2-320)*70*gain,((i%240)*2-240)*60*gain]).astype(np.int16)
        wavfile.write(inputs/'long.wav',48000,samples)
    if '--ranges' in sys.argv:
        wavfile.write(inputs/'oversized.wav',44100,np.tile(samples,(151,1)))
    os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
    provider=FileProvider()
    with LocalService(db,local_files_root=cache,reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe' if '--reaper' in sys.argv else None) as service:
        bridge=TaskBridge(provider,service)
        grants={role:provider.choose(role,lambda p=p:str(p)) for role,p in [('input',inputs),('output',saved)]}
        print(json.dumps({'ready':True,'out':str(out)},ensure_ascii=False),flush=True)
        for line in sys.stdin:
            try:
                request=json.loads(line);op=request['op']
                if op=='shutdown':break
                if op=='choose':value=grants[request['purpose']]
                elif op=='list':value=provider.list(grants['input']['id'])
                elif op=='read':
                    from hashlib import sha256
                    raw,digest=provider.read(request['id']);value={'base64':base64.b64encode(raw).decode(),'sha256':digest}
                else:value=bridge.invoke(request)
                print(json.dumps({'id':request['rpc_id'],'value':value},ensure_ascii=False),flush=True)
            except Exception as e:print(json.dumps({'id':request.get('rpc_id'),'error':str(e)},ensure_ascii=False),flush=True)
    provider.close()

if __name__=='__main__':main()
