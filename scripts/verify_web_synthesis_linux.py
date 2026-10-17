"""M06/M08 native receipts, fixed inputs and Windows numerical baselines."""
from pathlib import Path
import hashlib,io,json,struct,sys,tempfile
import numpy as np
import soundfile as sf
from ptb_worker.native.linux_runtime import load_profile,run
from ptb_worker.io.limits import Limits
from ptb_worker.m06_executor import unpack

def inputs():
    from phonetic_core.synthesis.klatt.api import defaults,generate,export_parameters
    c=defaults();c['duration']=1.;c['sequence']='aiueo'
    for curve in c['curves'].values():curve['points'][-1][0]=1.
    generated=generate(c)
    t=np.arange(16000)/16000;wav=io.BytesIO();sf.write(wav,.2*np.sin(2*np.pi*150*t),16000,format='WAV',subtype='FLOAT')
    return export_parameters(c),export_parameters(generated),wav.getvalue()

def stream_files(raw):
    files={};complete=False
    while raw:
        n=struct.unpack_from('<I',raw)[0];header=json.loads(raw[4:4+n]);raw=raw[4+n:]
        assert header['kind']!='error',header
        if header['kind']=='complete':assert header['count']==len(files) and not raw;complete=True;break
        size=header['size'];blob=raw[:size];raw=raw[size:];assert hashlib.sha256(blob).hexdigest()==header['sha256'];files[header['name']]=blob
    assert complete;return files

def main():
    out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False);profile_path,_=load_profile();profile_sha=hashlib.sha256(profile_path.read_bytes()).hexdigest();baseline=np.load(sys.argv[2],allow_pickle=False)
    parameters,generated,wav=inputs();checks={'m06':[],'m08':[]};observed={'m06':[],'m08':[]}
    with tempfile.TemporaryDirectory(prefix='synthesis-owned-',dir=out) as folder:
        folder=Path(folder);request=folder/'request.json';audio=folder/'source.wav';audio.write_bytes(wav);native=folder/'native.wav';native.touch()
        for action,params,source in [('generate',parameters,b''),('synthesize',generated,b''),('extract',parameters,wav)]:
            sha=hashlib.sha256(source).hexdigest();header=dict(parameters=params,action=action,seed=42,input_sha256=sha)
            request.write_bytes(json.dumps(header).encode()+b'\n'+source);e={};raw=run('m06',request,folder,Limits(input_bytes=1,output_bytes=24_000_000,process_bytes=1_000_000_000,timeout_seconds=120),evidence=e)
            files=dict(unpack(raw,sha,action));meta=json.loads(files['m06.ptb.json']);assert meta['action']==action
            if action=='synthesize':
                values,sr=sf.read(io.BytesIO(files['synthesis.wav']));assert sr==16000 and len(values)==16000;np.testing.assert_allclose(values,baseline['m06_audio'],rtol=0,atol=1/32768)
            if action=='generate':assert len(meta['config']['boundaries'])==4
            checks['m06'].append(action);observed['m06'].append(e)
        for action,config in [('preview',{'action':'preview'}),('transform',dict(action='transform',pitch_ratio=1.2,pitch_hz=20)),('synthesize',dict(action='synthesize',start=.1,end=.8,modified_f0=baseline['m08_f0'].tolist())),('linear',dict(action='linear',start=0,end=1,points=[dict(time=.1,freqs=[120,180],mode='full'),dict(time=.8,freqs=[140,200],mode='full')]))]:
            request.write_text(json.dumps(dict(config=config,sha256=hashlib.sha256(wav).hexdigest(),audio_path=str(audio),native_path=str(native),input_name='synthetic.wav')))
            e={};files=stream_files(run('m08',request,folder,Limits(input_bytes=1,output_bytes=64_000_000,process_bytes=1_000_000_000,timeout_seconds=120),evidence=e));meta=json.loads(files['m08.ptb.json'])
            np.testing.assert_allclose(meta['original_f0'],baseline['m08_f0'],atol=1e-5,rtol=1e-8)
            if action=='transform':
                value,sr=sf.read(io.BytesIO(files['synthetic.wav']));np.testing.assert_allclose(value,baseline['m08_transform'],atol=1/32768,rtol=0)
            if action=='synthesize':assert len(sf.read(io.BytesIO(next(v for n,v in files.items() if n.endswith('.wav'))))[0])==11200
            if action=='linear':assert len(files)==5
            assert native.stat().st_size==0;checks['m08'].append(action);observed['m08'].append(e)
    for phase in ('m06','m08'):
        report=dict(success=True,phase=phase,profile_sha256=profile_sha,checks=checks[phase],processes=observed[phase],owned_scratch_cleaned=True)
        (out/(phase+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(success=True,checks=checks)))
if __name__=='__main__':main()
