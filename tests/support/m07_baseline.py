"""Independent V2 capture: never imports phonetic_core. Public synthetic inputs only."""
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys
import types
import numpy as np
from scipy.io import wavfile
import scipy
import parselmouth

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Bypass package-wide eager GUI/EGG imports, not any M07 implementation.
for name in ('phonetic_toolbox','phonetic_toolbox.services','phonetic_toolbox.services.io','phonetic_toolbox.core','phonetic_toolbox.core.acoustic','phonetic_toolbox.core.manipulation','phonetic_toolbox.models'):
    package=types.ModuleType(name);package.__path__=[str(ROOT.joinpath(*name.split('.')))];sys.modules[name]=package
from phonetic_toolbox.services.phonation_synthesis_service import PhonationSynthesisService
from phonetic_toolbox.models.phonation_synthesis_models import PhonationAnalysisConfig, PhonationGenerationConfig, ContinuumType, F0AlignmentMode
from phonetic_toolbox.core.manipulation.phonation_synthesis import build_f0_control_axis, sample_f0_on_axis, interpolate_f0_control_points, enframe, build_lpc_residual, make_residual_continuum

OUT = ROOT/'tests/fixtures/m07'
EVIDENCE = ROOT/'output/validation/m07/baseline'

def capture(round_id):
    dest=EVIDENCE/round_id;dest.mkdir(parents=True,exist_ok=True)
    arrays={};cases=[];errors={};svc=PhonationSynthesisService()
    for index,(fs,duration,hz,amplitude,harmonic,pad) in enumerate([
        (8000,.37,120,.45,0,0), (16000,.51,175,.23,.15,.06),
        (11025,.293,135,.30,.06,0), (22050,.44,210,.58,.10,.04)]):
        t=np.arange(round(fs*duration))/fs
        audio=amplitude*np.sin(2*np.pi*hz*t)+harmonic*np.sin(4*np.pi*hz*t)
        audio=np.r_[np.zeros(round(fs*pad)),audio,np.zeros(round(fs*pad))]
        pcm=np.round(audio*32767).astype(np.int16);path=dest/f'input{index}.wav';wavfile.write(path,fs,pcm)
        arrays[f'input{index}']=pcm
        config=PhonationAnalysisConfig() if index<2 else PhonationAnalysisConfig(frame_length=160,frame_shift=40,lpc_order=16,preemphasis=.95,window_name='hann',trim_silence=False)
        before=svc._resample_audio(pcm.astype(float)/32768,fs,config.target_sample_rate)
        result=svc.analyze_file(path,config)
        arrays[f'{index}_resampled']=before
        for key in ('signal','f0_hz','lpc_coefficients','residual','pulses'):arrays[f'{index}_{key}']=getattr(result,key)
        arrays[f'{index}_bounds']=np.array([result.start_sample,result.end_sample])
        arrays[f'{index}_initial_f0']=svc._estimate_f0(result.signal,result.sample_rate,config)
        cases.append(dict(fs=fs,config=asdict(config)))
    pair=[svc.analyze_file(dest/f'input{i}.wav',PhonationAnalysisConfig()) for i in (0,1)]
    for mode in F0AlignmentMode:
        axis=build_f0_control_axis(pair[0].f0_hz,pair[1].f0_hz,21,mode);arrays[f'{mode.value}_axis']=axis
        for i,item in enumerate(pair):
            controls=sample_f0_on_axis(item.f0_hz,axis,mode)
            # Actual GUI renders three decimal F0 values before application.
            controls=np.round(controls,3);arrays[f'{mode.value}_{i}_controls']=controls
            arrays[f'{mode.value}_{i}_edit']=interpolate_f0_control_points(axis,controls,item.f0_hz,mode)
    for reverse in (False,True):
        a,b=pair[::-1] if reverse else pair
        for kind in ContinuumType:
            for energy in (False,True):
                for normalize in (False,True):
                    label=f'g_{int(reverse)}_{int(kind)}_{int(energy)}_{int(normalize)}'
                    gen=PhonationGenerationConfig(step_count=3,energy_match=energy,normalize_to_source=normalize,output_peak_limit=.83)
                    arrays[label+'_residual']=make_residual_continuum(a,b,kind,3,energy)
                    result=svc.generate_continuum(a,b,kind,gen);arrays[label]=result.audio_steps
                    exported=svc.export_continuum(result,dest/label)
                    for j,path in enumerate((*exported.step_files,exported.combined_file)):
                        arrays[label+f'_wav{j}']=np.frombuffer(path.read_bytes(),dtype=np.uint8)
    for n in (1,63,128,129,160,1000):
        audio=np.arange(n,dtype=float)/max(n,1);arrays[f'frames{n}']=enframe(audio,128,32)
        coeff,res=build_lpc_residual(audio,0,n-1,128,32,20)
        arrays[f'coeff{n}']=coeff;arrays[f'tail{n}']=res
    for name,values in [('silence',np.zeros(8000)),('short',np.ones(8)*.1),('constant',np.ones(4000)*.1)]:
        path=dest/f'{name}.wav';wavfile.write(path,8000,values.astype(np.float32))
        try:svc.analyze_file(path,PhonationAnalysisConfig())
        except Exception as exc:errors[name]=type(exc).__name__+':'+str(exc)
        else:errors[name]='accepted'
    hashes={}
    for rel in ['core/manipulation/phonation_synthesis.py','models/phonation_synthesis_models.py','services/phonation_synthesis_service.py','gui/widgets/phonation_synthesis_widget.py','gui/workers/phonation_synthesis_workers.py']:
        local=ROOT/'phonetic_toolbox'/rel;v2=ROOT.parent/'PhoneticToolbox_v2/phonetic_toolbox'/rel
        hashes[rel]=dict(inherited=hashlib.sha256(local.read_bytes()).hexdigest(),v2=hashlib.sha256(v2.read_bytes()).hexdigest())
    meta=dict(cases=cases,errors=errors,source_hashes=hashes,numpy=np.__version__,scipy=scipy.__version__,parselmouth=parselmouth.__version__,array_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()})
    return arrays,meta

if __name__=='__main__':
    first,meta=capture('round1');second,other=capture('round2')
    assert meta==other
    for key in first:np.testing.assert_array_equal(first[key],second[key])
    OUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/'v2.npz',**first)
    (OUT/'v2.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(arrays=len(first),values=sum(v.size for v in first.values()),double_run_exact=True,errors=meta['errors'],versions=[meta[k] for k in ('numpy','scipy','parselmouth')]),ensure_ascii=False))
