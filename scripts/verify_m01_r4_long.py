"""Synthetic long WAV through the real owned M01 child; retained local evidence."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import time
import sqlite3
from pathlib import Path
from uuid import uuid4
import numpy as np
import soundfile as sf
from ptb_worker.acoustic_stream_child import atomic
from ptb_worker.native.windows import OwnedProcess
from ptb_worker.process_entry import command
from ptb_api.acoustic_models import AcousticConfigSnapshot
from ptb_api.acoustic_extended import AcousticExtended,JointEggSettings
from ptb_worker.io.parameter_bundle import window,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=1800);p.add_argument('--full',action='store_true');p.add_argument('--derived',action='store_true');args=p.parse_args()
    repo=Path(__file__).resolve().parents[1];root=repo/'output/validation/m01-r4'/('long-'+uuid4().hex);root.mkdir(parents=True)
    rate=48000;block=np.arange(rate)/rate
    with sf.SoundFile(root/'input.wav','w',samplerate=rate,channels=2,subtype='PCM_16') as output:
        for second in range(args.seconds):
            t=block+second;phase=2*np.pi*(150*t+.3*np.sin(2*np.pi*.5*t))
            y=sum(.12/k*np.sin(k*phase) for k in range(1,16));egg=.6*np.sin(phase+.3)
            if second%37==18:y=y*0;egg=egg*0
            output.write(np.column_stack([egg,y]))
    config=AcousticConfigSnapshot(extended=AcousticExtended(audio_channel=1,egg=JointEggSettings(storage='cycles',derived=args.derived)))
    if not args.full:config.selection.keys=['pF0','rF0','Intensity']
    request=dict(config=config.model_dump(),index=0,audio_sha256=digest(root/'input.wav'),reaper_binary=str(repo/'phonetic_toolbox/core/acoustic/reaper.exe'))
    atomic(root/'request.json',request);start=time.monotonic()
    process=OwnedProcess(command('ptb_worker.acoustic_stream_child',str(root)),root,2_147_483_648)
    print(str(root),flush=True)
    try:
        while process.poll() is None:time.sleep(.5)
        response=json.loads((root/'response.json').read_text('utf-8'));assert response['success'],response
        conn=sqlite3.connect(root/'result.ptb.sqlite');meta=json.loads(conn.execute("SELECT value FROM ptb_metadata WHERE key='manifest'").fetchone()[0])
        times=np.fromiter((row[0] for row in conn.execute('SELECT Time_s FROM params ORDER BY Time_s')),float)
        assert len(times)>args.seconds*190 and np.allclose(np.diff(times),.005,atol=1e-10,rtol=0)
        assert meta['duration_s']==args.seconds
        for edge in [20,40,args.seconds-1]:
            if edge>=args.seconds:continue
            value=window(root/'result.ptb.sqlite','a'*64,dict(start=max(0.,edge-.1),end=min(float(args.seconds),edge+.1),width=1200,parameters=['F0 - Praat','F0 - GCI']))
            assert value['tracks']['F0 - Praat'] and value['tracks']['F0 - GCI']
        report=dict(success=True,seconds=args.seconds,rate=rate,channels=2,full=args.full,derived=args.derived,
            wall_seconds=time.monotonic()-start,memory_peak_bytes=process.memory_peak(),rows={k:v['rows'] for k,v in meta['tables'].items()},columns=meta['tables']['params']['columns'],
            files={n:(root/n).stat().st_size for n in response['files']},max_frame_gap_s=float(np.max(np.diff(times))))
        conn.close()
    finally:
        process.close()
    report['group_cleaned']=process.group_cleaned;atomic(root/'verification.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
