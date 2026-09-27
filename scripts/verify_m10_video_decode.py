"""Independent decoder QA; installed FFmpeg is a test tool, never packaged."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate,correlation_lags
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.animation import prepare_animation
from ptb_desktop.vocal_tract.audio_output import audition_samples

ROOT=Path(__file__).resolve().parents[1]
out=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'output/validation/m10/r4-qt'
result=json.loads((out/'result.json').read_text('utf-8'));reports=[]
cases=[('current',(1280,720)),('six',(1920,1080))] if (out/'current.webm').exists() else [('ordered-current',(1280,720))]
for name,size in cases:
    seqfile=out/('ordered-sequence.json' if name=='ordered-current' else name+'-sequence.json')
    sequence=json.loads((seqfile if seqfile.exists() else Path(result['profile'])/'keyframes.json').read_text('utf-8'))
    e=Engine(resource_dir=ROOT/'resources/vocal_tract/native')
    try:prepared=prepare_animation(e,sequence['frames'],pitch_curve=sequence['pitch_curve'],pictures_enabled=False)
    finally:e.close()
    reference=audition_samples(prepared['audition_audio'],.8,0,prepared['envelope'])
    path=out/(name+'.webm')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
    v=next(s for s in probe['streams'] if s['codec_type']=='video');a=next(s for s in probe['streams'] if s['codec_type']=='audio')
    assert (v['width'],v['height'])==size and v['codec_name']=='vp8' and a['codec_name']=='opus'
    assert abs(float(probe['format']['duration'])-prepared['duration'])<1e-6
    packets=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v','-show_packets','-of','json',str(path)]))['packets']
    assert len(packets)==int(np.ceil(prepared['duration']*30))
    assert all(abs(float(p['pts_time'])-i/30)<=.000501 for i,p in enumerate(packets))
    assert abs(float(packets[-1]['pts_time'])+float(packets[-1]['duration_time'])-prepared['duration'])<=.001
    wav=out/(name+'-decoded.wav');subprocess.run(['ffmpeg','-v','error','-i',str(path),'-map','0:a:0','-c:a','pcm_f32le','-y',str(wav)],check=True)
    sr,audio=wavfile.read(wav);assert sr==48000 and len(audio)==len(reference),(len(audio),len(reference))
    corr=correlate(audio,reference,mode='full',method='fft');lags=correlation_lags(len(audio),len(reference));lag=int(lags[np.argmax(corr)])
    assert abs(lag)<=24,lag
    similarity=float(np.dot(audio,reference)/(np.linalg.norm(audio)*np.linalg.norm(reference)))
    assert similarity>.95,similarity
    subprocess.run(['ffmpeg','-v','error','-i',str(path),'-ss',str(prepared['duration']/2),'-frames:v','1','-y',str(out/(name+'-decoded.png'))],check=True)
    reports.append(dict(name=name,frames=len(packets),duration=prepared['duration'],samples=len(audio),lag_samples=lag,correlation=similarity))
(out/'decode.json').write_text(json.dumps(reports,indent=2),encoding='utf-8');print(json.dumps(reports,indent=2))
