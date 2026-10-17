"""Synthetic-only real bounded preview cache, owner/config and monitor parity."""
from pathlib import Path
import hashlib,io,json,sqlite3,sys,time
import numpy as np
from scipy.io import wavfile
from ptb_worker.native.linux_runtime import load_profile
from ptb_worker import spectrogram_preview,recording_preview,parameter_preview,spec2wav_preview
from ptb_worker.preview_cache import preview_scope,cache


def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    path,_=load_profile();records=[]
    def measure(module,case,call):
        started=time.monotonic();value=call();elapsed=time.monotonic()-started
        encoded=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
        records.append(dict(module=module,case=case,seconds=elapsed,bytes=len(encoded),sha256=hashlib.sha256(encoded).hexdigest()))
        return value
    fs=16000;t=np.arange(fs*2)/fs;w=io.BytesIO();wavfile.write(w,fs,np.column_stack([.2*np.sin(t*2*np.pi*150),.15*np.sin(t*2*np.pi*230)]).astype('float32'));raw=w.getvalue()
    with sqlite3.connect(':memory:') as c:
        c.execute('CREATE TABLE params(Time_s REAL,F0_Hz REAL)');c.executemany('INSERT INTO params VALUES(?,?)',[(0.,150.),(.01,None),(.02,155.)]);c.commit();table=c.serialize()
    from PIL import Image
    y,x=np.indices((96,160));gray=np.uint8(np.clip(255-150*np.exp(-((y-35-5*np.sin(x/18))/5)**2)-80*np.exp(-((y-65)/8)**2),0,255));w=io.BytesIO();Image.fromarray(gray).save(w,format='PNG');png=w.getvalue()
    base=dict(mode='image',time_start=0,time_end=.6,freq_start=0,freq_end=5000,n_iter=32,reconstruction_profile='calibrated',phase_initialization='auto',frequency_axis='linear',auto_parameters=True,win_length_ms=10,min_db=-60,max_db=0,target_sr=0,seed=0)
    cases=[('shared_spectrogram',lambda:spectrogram_preview.render(raw,channel=0,start=0,end=2,width=640),'4a7ac0603117e46bdceda7b112a1b0482c7394bc389b571b550a40897ab50081'),('recording_spectrogram',lambda:recording_preview.render(raw,start=0,end=fs*2,channel=0,width=640),'b964b11a9c57fbfe6d5d0a873273e338a667b44605128345379da1dd824474b9'),('parameter_table',lambda:parameter_preview.render(table,'test.ptb.sqlite'),'219ba63762a14dc6053ef25412c466bacb9e841ff3744fd64ee2cfbd2f3b8c0b'),('spec2wav_prediction',lambda:spec2wav_preview.render(png,base),'1cb60f67d23217e8e08a03a4676654e37e565e2fd3f452daaee0fd28e342d7b0')]
    baselines={}
    for name,call,expected in cases:
        with preview_scope(('synthetic-a',name)):
            a=measure(name,'first',call);baselines[name]=a;b=measure(name,'cache_hit',call);assert a==b and records[-1]['sha256']==expected
            # A hit still validates all registered runtime file identities.
            assert records[-1]['seconds']<records[-2]['seconds']
        with preview_scope(('synthetic-b',name)):
            c=measure(name,'other_owner',call);assert c==a
    assert cache.bytes<=cache.max_bytes and len(cache.items)<=cache.max_entries
    with preview_scope(('synthetic-a','shared_spectrogram')):
        changed=measure('shared_spectrogram','different_channel',lambda:spectrogram_preview.render(raw,channel=1,start=0,end=2,width=640));assert changed['pixels_base64']!=baselines['shared_spectrogram']['pixels_base64']
    from phonetic_core.vocal_tract.monitor import OutputHistory
    from phonetic_core.vocal_tract import session
    from types import SimpleNamespace
    history=OutputHistory(48000);history.append(np.sin(np.arange(24000)*2*np.pi*150/48000).astype('float32'))
    runtime=session.Runtime(SimpleNamespace(section_count=33,sr=48000),None,SimpleNamespace(output_history=history))
    first=measure('vocal_monitor','first',lambda:runtime.invoke('audio/monitor'));repeat=measure('vocal_monitor','cache_hit',lambda:runtime.invoke('audio/monitor'));assert first==repeat
    history.reset();assert runtime.invoke('audio/monitor')['generation']!=first['generation']
    report=dict(success=True,profile_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=records,checks=['four_fixed9_json_sha_baselines_exact','owner_isolation','channel_invalidation','bounded_ram','runtime_verified_on_cache_hit','monitor_output_parity','monitor_generation_invalidation'],synthetic_only=True)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
