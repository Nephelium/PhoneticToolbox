"""Capture public fixture computations under the unchanged current source.

Run separately on Windows and macOS. This is a port comparison, not a replacement
for independently captured v2 tests. No native input devices or user files.
"""
import argparse,json,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in ('backend/src','packages/phonetic_core/src','desktop/src'):sys.path.insert(0,str(ROOT/p))
_dll_handle=None
if sys.platform=='win32' and (Path(sys.prefix)/'Library/bin').is_dir():
    _dll_handle=os.add_dll_directory(str(Path(sys.prefix)/'Library/bin'))
import numpy as np

def capture(phase):
    values={};meta={}
    if phase=='egg':
        from phonetic_core.egg import EGGConfig,prepare,analyze_events
        from phonetic_core.egg.metrics import calculate_cq_sq
        from phonetic_core.lpc import LPCConfig,compute_spectrum
        data=np.load(ROOT/'tests/fixtures/m03/EGG-SYN-PCM16.npz')
        spec=json.loads((ROOT/'tests/fixtures/m03/EGG-SYN-PCM16.json').read_text('utf8'))
        samples=np.column_stack([data['load.egg_signal_raw'],data['load.audio_signal']])
        for i,variant in enumerate(spec['variants']):
            config=EGGConfig(**variant['config']);result=analyze_events(prepare(samples,44100,config),config)
            for name in ('egg_signal_processed','gci_times','goi_times','peak_times'):
                values['egg.'+str(i)+'.'+name]=np.asarray(getattr(result,name),dtype=float)
            for j,v in enumerate(calculate_cq_sq(result.gci_times,result.goi_times,result.peak_times)):
                if v is not None:values['egg.'+str(i)+'.cq'+str(j)]=np.asarray(v,dtype=float)
        data=np.load(ROOT/'tests/fixtures/m04/arrays.npz');cases=json.loads((ROOT/'tests/fixtures/m04/result.json').read_text('utf8'))['cases']
        for name,case in cases.items():
            if case['status']!='error':
                result=compute_spectrum(data[name+'.input'],case['rate'],LPCConfig(**case['config']))
                values['lpc.'+name+'.frequency']=result.frequencies_hz;values['lpc.'+name+'.db']=result.magnitude_db
    else:
        from phonetic_core.synthesis.klatt.api import defaults,generate,synthesize
        for i,sequence in enumerate(('a','a i u')):
            c=defaults();c['sequence']=sequence;c=generate(c);np.random.seed(1234+i)
            values['m06.'+str(i)+'.audio']=synthesize(c)
            for name,curve in c['curves'].items():values['m06.'+str(i)+'.curve.'+name]=np.array(curve['points'],float)
        import parselmouth
        from phonetic_core.manipulation.m08_rules import track
        from phonetic_core.manipulation.m08_synthesis import synthesize_from_pitch
        data=np.load(ROOT/'tests/fixtures/m08/v2.npz');spec=json.loads((ROOT/'tests/fixtures/m08/v2.json').read_text('utf8'))
        parselmouth.praat.run('random_initializeWithSeedUnsafelyButPredictably (42)')
        sound=parselmouth.Sound(data['samples'],spec['sample_rate']);times,f0=track(sound)
        values['m08.times']=times;values['m08.f0']=f0
        for i,(start,end,speed) in enumerate(spec['synth']):values['m08.audio'+str(i)]=synthesize_from_pitch(sound,data['times'],data['f0']*1.2,start,end,speed).values
        from phonetic_core.manipulation import m07_api as core
        from phonetic_core.manipulation.m07_models import PhonationAnalysisConfig,PhonationGenerationConfig
        from ptb_worker.m07_science import estimator
        data=np.load(ROOT/'tests/fixtures/m07/v2.npz');spec=json.loads((ROOT/'tests/fixtures/m07/v2.json').read_text('utf8'))
        analyses=[core.analyze(data['input'+str(i)].astype(float)/32768,c['fs'],PhonationAnalysisConfig(**c['config']),estimator()) for i,c in enumerate(spec['cases'])]
        for i,item in enumerate(analyses):
            for name in ('signal','residual','lpc_coefficients','f0_hz','pulses'):values['m07.'+str(i)+'.'+name]=getattr(item,name)
        for kind in (1,2,3):values['m07.audio'+str(kind)]=core.generate(*analyses[:2],kind,PhonationGenerationConfig(3,True,True,.83)).audio_steps
        from phonetic_core.spec2wav import reconstruct
        gray=(np.arange(17*23,dtype=np.uint16).reshape(17,23)*7%256).astype('uint8')
        result=reconstruct(gray,time_end=.3,freq_end=4000,target_sr=16000,n_iter=32,seed=42)
        values['m09.audio']=result['audio'];meta['m09.sr']=result['sr']
        from phonetic_core.services.acoustic import analyze_audio
        from phonetic_core.models.acoustic import AcousticConfig
        from phonetic_core.models.audio import AudioInput
        data=np.load(ROOT/'tests/fixtures/m03/EGG-SYN-PCM16.npz')
        frame=analyze_audio(AudioInput(data['load.audio_signal'],44100),AcousticConfig(use_reaper=False,only_voiced=False)).to_dataframe()
        for name in frame.select_dtypes(include='number').columns:values['m01.'+name]=frame[name].to_numpy()
    return values,meta

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['host','egg'],required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from macos_validation_identity import source_hashes,require_identity,sha256
    identity=source_hashes(ROOT)
    values,meta=capture(args.phase)
    require_identity(source_hashes(ROOT),identity,'Capture inputs changed during verification')
    np.savez_compressed(args.output/(args.phase+'.npz'),**values)
    meta.update(platform=sys.platform,phase=args.phase,arrays=len(values),source_sha256=identity,capture_sha256=sha256(Path(__file__)),npz_sha256=sha256(args.output/(args.phase+'.npz')))
    (args.output/(args.phase+'.json')).write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8');print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
