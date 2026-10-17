"""Bounded Linux M09 science admission with owned synthetic images/audio only."""
from pathlib import Path
import hashlib,io,json,sys,tempfile
import numpy as np
import cv2
import soundfile as sf
from ptb_worker.native.linux_runtime import load_profile,run
from ptb_worker.io.limits import Limits
from ptb_worker.segmentation import unpack_bundle
from ptb_worker.spec2wav_preview import render
from ptb_worker.spectrogram_preview import PreviewError

def fixture():
    y,x=np.indices((96,160));gray=np.uint8(np.clip(255-150*np.exp(-((y-35-5*np.sin(x/18))/5)**2)-80*np.exp(-((y-65)/8)**2),0,255))
    return gray,cv2.imencode('.png',gray)[1].tobytes()

def main():
    out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    profile_path,_=load_profile();gray,png=fixture();checks=[];evidence=[]
    base=dict(mode='image',time_start=0,time_end=.6,freq_start=0,freq_end=5000,n_iter=32,reconstruction_profile='calibrated',phase_initialization='auto',frequency_axis='linear',auto_parameters=False,win_length_ms=10,min_db=-60,max_db=0,target_sr=0,seed=0)
    with tempfile.TemporaryDirectory(prefix='m09-owned-',dir=out) as folder:
        folder=Path(folder)
        predicted=render(png,{**base,'auto_parameters':True});assert predicted['source_sha256']==hashlib.sha256(png).hexdigest()
        assert np.isfinite(list(v for v in predicted['prediction'].values() if type(v) in (int,float))).all();checks.append('builtin_prediction')
        for label,config,raw in [('image',base,png),('four_corners',{**base,'corners':[dict(x=.1,y=.1),dict(x=.9,y=.1),dict(x=.9,y=.9),dict(x=.1,y=.9)]},png),('image_draw',{**base,'mode':'image_draw','strokes':[dict(points=[dict(x=.2,y=.5),dict(x=.8,y=.5)],size=5,size_unit='px',opacity=.5,color=0)]},png)]:
            request=folder/'request.json';request.write_bytes(json.dumps(dict(config=config,sha256=hashlib.sha256(raw).hexdigest())).encode()+b'\n'+raw)
            observed={};payload=run('spec2wav',request,folder,Limits(input_bytes=1,output_bytes=64_000_000,process_bytes=1_000_000_000,timeout_seconds=120),evidence=observed)
            bundle=unpack_bundle(payload,64_000_000);files=dict(zip([f['name'] for f in bundle.manifest['files']],bundle.payloads))
            assert set(files)=={'calibrated.png','reconstructed.png','reconstructed.wav','reconstruction.ptb.json'}
            audio,sr=sf.read(io.BytesIO(files['reconstructed.wav']));meta=json.loads(files['reconstruction.ptb.json'])
            assert sr==10000 and len(audio)==6000 and np.isfinite(audio).all() and np.max(np.abs(audio))>0 and meta['source_sha256']==hashlib.sha256(raw).hexdigest()
            if label=='image':
                baseline=Path(sys.argv[2]);expected=np.load(baseline,allow_pickle=False)
                np.testing.assert_allclose(audio,expected['audio'],atol=1/32768,rtol=0)
                np.testing.assert_allclose([predicted['prediction'][k] for k in ('predicted_window_ms','dynamic_range_db')],expected['prediction'],atol=.0001,rtol=0)
            checks.append(label);evidence.append(observed)
        x=np.column_stack([np.sin(np.arange(8000)*2*np.pi*220/16000)*.1,np.linspace(-.1,.1,8000)])
        wav=io.BytesIO();sf.write(wav,x,16000,format='WAV',subtype='FLOAT');raw=wav.getvalue();config={**base,'mode':'audio_draw','channel':0,'audio_fft':1024,'strokes':[]}
        request.write_bytes(json.dumps(dict(config=config,sha256=hashlib.sha256(raw).hexdigest())).encode()+b'\n'+raw)
        observed={};bundle=unpack_bundle(run('spec2wav',request,folder,Limits(input_bytes=1,output_bytes=64_000_000,process_bytes=1_000_000_000,timeout_seconds=120),evidence=observed),64_000_000)
        files=dict(zip([f['name'] for f in bundle.manifest['files']],bundle.payloads));audio,sr=sf.read(io.BytesIO(files['reconstructed.wav']));assert sr==16000 and audio.shape==x.shape
        np.testing.assert_array_equal(audio[:,1],x[:,1].astype('float32'));checks.append('audio_draw_other_channel_exact');evidence.append(observed)
        try:render(png,base,timeout=.01)
        except PreviewError as e:assert e.code=='preview_timeout'
        else:raise AssertionError('Timeout did not enforce process budget')
        assert render(png,base)['width']==160;checks.append('timeout_cleanup_and_next_preview')
    report=dict(success=True,phase='spec2wav',profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),checks=checks,processes=evidence,owned_scratch_cleaned=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(success=True,checks=checks)))
if __name__=='__main__':main()
