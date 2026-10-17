"""Fixed synthetic M16 native admission, no user accounts or user recordings."""
import base64,hashlib,io,json,os,struct,sys,tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from ptb_worker.native.linux_runtime import run,load_profile,command
from ptb_worker.native.posix import run_bounded
from ptb_worker.io.limits import Limits

def main():
    output=Path(sys.argv[1]).resolve();output.mkdir(parents=True,exist_ok=False)
    profile_path,profile=load_profile();checks=[];evidence=[]
    report={'phase':'recording','profile_sha256':hashlib.sha256(profile_path.read_bytes()).hexdigest(),'success':False}
    with tempfile.TemporaryDirectory(prefix='m16-owned-',dir=output) as folder:
        folder=Path(folder);rng=np.random.default_rng(612);n=270000
        x=np.column_stack([rng.normal(0,.02,n),np.linspace(-.3,.3,n)]).astype(np.float32)
        x[8000:,0]+=(np.sin(np.arange(n-8000)*2*np.pi*220/48000)*.2).astype(np.float32)
        source=folder/'source.wav';sf.write(source,x,48000,format='WAV',subtype='FLOAT')
        base=dict(roles=['microphone','egg'],channels=[0],channel=0,gain_db=6.,strength=1.,subtype='FLOAT')
        for action,extra in [('gain',{}),('keep',{'start':100,'end':5000}),('noise',{'start':0,'end':5000}),('denoise',{'reference_start':0,'reference_end':5000}),('export',{'subtype':'PCM_24','start':100,'end':5000})]:
            request=folder/'request.json';request.write_text(json.dumps({'paths':[str(source)],'config':dict(base,action=action,**extra)}))
            observed={};raw=run('recording',request,folder,Limits(input_bytes=1,output_bytes=10_000_000,process_bytes=800_000_000,timeout_seconds=60),evidence=observed)
            size=struct.unpack_from('<I',raw)[0];header=json.loads(raw[4:4+size]);assert header['ok'],header
            if action!='noise':
                y,rate=sf.read(io.BytesIO(raw[4+size:]),dtype='float32',always_2d=True);assert rate==48000
                if action=='gain':np.testing.assert_allclose(y[:,0],x[:,0]*10**(.3),rtol=2e-7,atol=0);np.testing.assert_array_equal(y[:,1],x[:,1])
                if action in ('keep','export'):np.testing.assert_allclose(y,x[100:5000],atol=2**-23,rtol=0)
                if action=='denoise':assert y.shape==x.shape;np.testing.assert_array_equal(y[:,1],x[:,1]);assert np.mean(y[:5000,0]**2)<np.mean(x[:5000,0]**2)
            checks.append(action);evidence.append(observed)
        payload=source.read_bytes();header=json.dumps(dict(size=len(payload),start=100,end=15000,channel=0,width=320)).encode()+b'\n';observed={}
        raw=run_bounded(command('recording_preview','--preview'),header+payload,folder,Limits(input_bytes=10_000_000,output_bytes=512000,process_bytes=800_000_000,timeout_seconds=60),evidence=observed)
        value=json.loads(raw);assert value['backend']=='praat' and value['display_revision']=='m16-praat-display/1';assert len(base64.b64decode(value['pixels_base64']))==value['columns']*len(value['frequencies']);checks.append('praat');evidence.append(observed)
    report.update(success=True,checks=checks,processes=evidence,owned_scratch_cleaned=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'success':True,'checks':checks}))

if __name__=='__main__':main()
