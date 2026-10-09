"""M03-R7 real isolated runtime, public synthetic input and measured resources."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
from verification_artifacts import verification_run
import time
from uuid import uuid4
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from ptb_worker.egg_interactive import InteractivePreview
from ptb_worker.egg_runtime import command
from ptb_worker.native import windows as win

ROOT=Path(__file__).resolve().parents[1]


def memory(manager):
    limits=win.ExtendedLimit()
    query=win.api('QueryInformationJobObject',[win.w.HANDLE,win.c.c_int,win.c.c_void_p,win.w.DWORD,win.c.c_void_p])
    win.checked(query(manager.job,9,win.c.byref(limits),win.c.sizeof(limits),None))
    return int(limits.peak_job)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);args=parser.parse_args()
    with verification_run('m03-r7', 'native', '48 kHz stereo PCM16, 1800 s; deterministic 150 Hz harmonics, 20 s blocks, amplitude .2 then .5; external --source is retained') as (out, scratch):
        path=args.source or scratch/'synthetic-30min.wav';fs=48000
        if not args.source:
          with sf.SoundFile(path,'w',samplerate=fs,channels=2,subtype='PCM_16') as f:
            for first in range(0,1800,20):
                t=(np.arange(fs*20)+first*fs)/fs
                scale=.2 if first<900 else .5
                egg=scale*(np.sin(2*np.pi*150*t)+.2*np.sin(2*np.pi*300*t))
                audio=scale*(np.sin(2*np.pi*150*t)+.1*np.sin(2*np.pi*450*t))
                f.write(np.column_stack((egg,audio)))
        os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
        manager=InteractivePreview(reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe')
        report=dict(success=False,measurements=[],source=str(path),schema_applied=[])
        try:
            clock=time.perf_counter();opened=manager.open('test',b'',source_path=path)
            report['open_seconds']=time.perf_counter()-clock
            assert opened['sample_count']==fs*1800 and opened['sample_rate_hz']==fs and opened['overview_base64']
            config=dict(mode='preview',roi_start=1790,roi_end=1800,micro_center=1795,keep_gci_f0=False,keep_praat_f0=False,f0_policy='audio-f0/2')
            for label,changes in [('initial',{}),('micro',dict(micro_center=1795.01)),('filter',dict(highpass_cutoff=30)),
                ('sixty-second',dict(roi_start=1720,roi_end=1780,micro_center=1750)),
                ('all-f0',dict(keep_gci_f0=True,keep_praat_f0=True,keep_reaper_f0=True)),
                ('warm-f0-micro',dict(micro_center=1750.01)),('warm-f0-filter',dict(highpass_cutoff=35)),('moved-f0-view',dict(roi_start=100,roi_end=110,micro_center=105))]:
                config.update(changes);clock=time.perf_counter()
                result=manager.update('test',opened['session_id'],config)
                assert result['sample_count']==fs*1800
                assert result['preview']['cq']['times'] and max(result['preview']['cq']['times'])>config['roi_start']
                if result['audio_base64']:
                    rate,audio=wavfile.read(io.BytesIO(base64.b64decode(result['audio_base64'])))
                    assert rate==fs and len(audio)==int((config['roi_end']-config['roi_start'])*fs)
                    assert result['audio_start_s']==config['roi_start']
                report['measurements'].append(dict(label=label,seconds=time.perf_counter()-clock,
                    bytes=memory(manager),cq_points=len(result['preview']['cq']['times']),
                    f0_points=len(result['preview']['praat']['times'])))
                (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            report['success']=True
        finally:
            pid=manager.process.pid if manager.process else None
            manager.close();report['process_released']=manager.process is None;report['pid']=pid
            (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(out,flush=True)

if __name__=='__main__':main()
