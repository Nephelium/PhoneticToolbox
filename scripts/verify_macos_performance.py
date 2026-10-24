"""Actual Mac previews, exact result comparison and repeated-work measurements.

Synthetic inputs only. No physical microphone, speaker or GPU timing claims.
"""
import hashlib,io,json,os,sqlite3,statistics,subprocess,sys,time
from pathlib import Path

def main(output,phase='after'):
    import numpy as np
    import soundfile as sf
    import cv2
    import psutil
    from ptb_worker.native import darwin
    from ptb_worker.spectrogram_session import SpectrogramSession
    from ptb_worker.parameter_preview import render as parameters
    from ptb_worker.spec2wav_preview import render as image_preview
    from ptb_desktop.recording.capture import preview_array
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(success=False,phase=phase,frozen=bool(getattr(sys,'frozen',False)),scope='Actual macOS native children and desktop code; synthetic inputs; no physical-device or web latency claim',cases={},checks=[])
    def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
    def timed(name,fn,repeat=3):
        rows=[];values=[]
        for _ in range(repeat):
            start=time.perf_counter();value=fn();rows.append(time.perf_counter()-start);values.append(digest(value))
        assert len(set(values))==1,(name,'nondeterministic')
        report['cases'][name]=dict(seconds=rows,median_seconds=statistics.median(rows),sha256=values[0])
        print(name,[round(x,6) for x in rows],flush=True)
        return value
    spawned=[];original_popen=subprocess.Popen
    def observed_popen(*args,**kwargs):
        process=original_popen(*args,**kwargs)
        if kwargs.get('start_new_session'):spawned.append(process.pid)
        return process
    subprocess.Popen=observed_popen
    session=SpectrogramSession(idle_seconds=1)
    try:
        sr=44100;t=np.arange(round(7.3*sr))/sr
        audio=np.column_stack([.2*np.sin(2*np.pi*150*t),.3*np.sin(2*np.pi*230*t)])
        f=io.BytesIO();sf.write(f,audio,sr,format='WAV',subtype='FLOAT');raw=f.getvalue()
        base=dict(channel=0,start=0.,end=7.3,width=800)
        before=len(spawned)
        original=timed('spectrogram_same',lambda:session.render(raw,**base))
        for name,kwargs in [('spectrogram_roi',{**base,'start':1.,'end':2.}),('spectrogram_channel',{**base,'channel':1}),('spectrogram_width',{**base,'width':500})]:
            value=timed(name,lambda k=kwargs:session.render(raw,**k),2)
            assert digest(value)!=digest(original)
        report['spectrogram_spawned']=len(spawned)-before
        assert digest(session.render(raw,**base))==digest(original)
        session.close();assert all(not darwin.group_members(pid) for pid in spawned[before:])
        if phase!='before':
            # Malformed input must close only the owned process, then permit recovery.
            try:session.render(b'invalid wave',**base)
            except Exception:pass
            else:raise AssertionError('Invalid WAV accepted')
            assert digest(session.render(raw,**base))==digest(original)
            pid=session._darwin.process.pid
            time.sleep(1.3);assert session._darwin.process is None and not darwin.group_members(pid)
        report['checks'].append('spectrogram_exact_repeat_roi_channel_width_and_owned_close')
        # SQLite bytes exercise the actual legacy table decoder.
        db=sqlite3.connect(':memory:');db.execute('CREATE TABLE params(Time_s REAL,F0 REAL,textgrid TEXT)')
        db.executemany('INSERT INTO params VALUES(?,?,?)',[(i*.01,150+i%7,'a') for i in range(2000)])
        db.commit();table=db.serialize();db.close()
        timed('parameters_same',lambda:parameters(table,'qa.ptb.sqlite'))
        db=sqlite3.connect(':memory:');db.deserialize(table);db.execute('UPDATE params SET F0=240 WHERE rowid=1');db.commit();changed=db.serialize();db.close()
        assert digest(parameters(changed,'qa.ptb.sqlite'))!=report['cases']['parameters_same']['sha256']
        try:parameters(table,'qa.txt')
        except Exception:pass
        else:raise AssertionError('Filename format must participate in cache identity')
        report['checks'].append('parameter_content_and_filename_invalidation')
        y,x=np.indices((96,160));gray=np.uint8(np.clip(255-150*np.exp(-((y-35-5*np.sin(x/18))/5)**2),0,255));png=cv2.imencode('.png',gray)[1].tobytes()
        config=dict(mode='image',time_start=0,time_end=.3,freq_start=0,freq_end=5000,n_iter=32,reconstruction_profile='calibrated',phase_initialization='auto',frequency_axis='linear',auto_parameters=True,win_length_ms=10,min_db=-60,max_db=0,target_sr=0,seed=0)
        value=timed('image_prediction_same',lambda:image_preview(png,config))
        assert digest(image_preview(png,{**config,'invert_image':True}))!=digest(value)
        assert digest(image_preview(png,{**config,'auto_parameters':False}))!=digest(value)
        try:image_preview(b'not an image',config)
        except Exception:pass
        else:raise AssertionError('Invalid image accepted')
        assert digest(image_preview(png,config))==digest(value)
        report['checks'].append('image_content_settings_error_and_recovery')
        from ptb_desktop.vocal_tract.runtime import Runtime
        root=darwin.bundle_root();runtime=Runtime(root/'resources/vocal_tract/native',output/'vocal-profile',playback_allowed=False)
        try:
            samples=.2*np.sin(2*np.pi*150*np.arange(144000)/48000)
            runtime.live.output_history.append(samples)
            m=timed('vocal_monitor_same',lambda:runtime.invoke('audio/monitor',dict(seconds=3,window_ms=20,hop_ms=10)))
            runtime.live.output_history.append(samples[:960])
            assert digest(runtime.invoke('audio/monitor',dict(seconds=3,window_ms=20,hop_ms=10)))!=digest(m)
            assert digest(runtime.invoke('audio/monitor',dict(seconds=3,window_ms=40,hop_ms=10)))!=digest(m)
            runtime.live.output_history.reset();assert runtime.invoke('audio/monitor')['duration']==0
            report['checks'].append('output_monitor_sample_generation_settings_invalidation')
        finally:runtime.close()
        samples=np.column_stack([.2*np.sin(2*np.pi*150*np.arange(144000)/48000)]*2).astype(np.float32)
        timed('recording_waveform',lambda:preview_array(samples,48000,False))
        timed('recording_waveform_spectrum',lambda:preview_array(samples,48000,True))
        if phase!='before':
            from ptb_worker.native.darwin_preview_cache import cache
            # Mutating a returned object cannot poison later results.
            value=image_preview(png,config);expected=digest(value);value['width']=-1
            assert digest(image_preview(png,config))==expected
            # Bounded cache, concurrent identical requests, failures are retried.
            from ptb_worker.native.darwin_preview_cache import ResultCache
            from concurrent.futures import ThreadPoolExecutor
            c=ResultCache(max_bytes=100,max_entries=2,ttl=.05);calls=[]
            def compute():calls.append(1);time.sleep(.02);return {'ok':1}
            with ThreadPoolExecutor(4) as pool:assert list(pool.map(lambda _:c.get('same',compute),range(4)))==[{'ok':1}]*4
            assert len(calls)==1
            time.sleep(.06);c.get('same',compute);assert len(calls)==2
            c.get('big',lambda:{'x':'z'*200});assert c.bytes<=100
            for _ in range(2):
                try:c.get('bad',lambda:1/0)
                except ZeroDivisionError:pass
                else:raise AssertionError('Error cached')
            assert c.stats()['entries']<=2
            report['cache_stats']=cache.stats();report['checks'].append('cache_budgets_ttl_singleflight_errors_and_detached_results')
            from ptb_worker.m05_executor import sampled_disk_bytes
            import threading
            scratch=output/'atomic-stat';scratch.mkdir();stop=threading.Event();written=[]
            def publish():
                while not stop.is_set():
                    (scratch/'status.next').write_bytes(b'x'*4096)
                    os.replace(scratch/'status.next',scratch/'status.json');written.append(1)
            thread=threading.Thread(target=publish);thread.start()
            try:
                for _ in range(10000):assert 0<=sampled_disk_bytes(scratch)<=8192
            finally:stop.set();thread.join()
            report['atomic_stat_stress']=dict(samples=10000,atomic_replacements=len(written))
            report['checks'].append('m05_disk_sampling_during_real_atomic_replacements')
        report['spawned_pids']=spawned
        report['rss_bytes']=psutil.Process().memory_info().rss
        report['success']=True
    finally:
        session.close();subprocess.Popen=original_popen
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main(*sys.argv[1:]))
