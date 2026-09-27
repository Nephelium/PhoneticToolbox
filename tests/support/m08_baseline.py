"""Capture independent V2 M08 outputs. Never imports the V3 implementation."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
import parselmouth

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'phonetic_toolbox'
OUT = ROOT / 'tests/fixtures/m08'


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture():
    synth = load('v2_synth', 'core/manipulation/synthesis.py')
    batch = load('v2_batch', 'core/manipulation/batch_utils.py')
    # AST extract the exact method, keeping original file read/save operations.
    source = (SOURCE/'services/manipulation_service.py').read_text(encoding='utf8')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='process_single_file')
    module = ast.Module(body=[method],type_ignores=[])
    namespace = {'os':__import__('os'), 'parselmouth':parselmouth,'call':parselmouth.praat.call}
    exec(compile(module,'v2_process_single_file','exec'),namespace)
    sr=16000; t=np.arange(9600)/sr
    samples=.3*np.sin(2*np.pi*150*t)+.08*np.sin(2*np.pi*300*t)
    snd=parselmouth.Sound(samples,sr);pitch=snd.to_pitch();times=pitch.xs();f0=pitch.selected_array['frequency']
    data=dict(samples=samples,times=times,f0=f0)
    cases=[]
    for i,(start,end,speed) in enumerate([(0,.6,1),(.1,.5,.8),(.1,.5,1.5),(0,.6,1.005)]):
        parselmouth.praat.run("random_initializeWithSeedUnsafelyButPredictably (42)")
        curve=f0*1.2;out=synth.synthesize_from_pitch(snd,times,curve,start,end,speed)
        data[f'synth_{i}']=out.values;data[f'axis_{i}']=np.array([out.xmin,out.xmax,out.dx,out.x1]);cases.append([start,end,speed])
    batch_cases=[]; errors={}
    with tempfile.TemporaryDirectory(prefix='ptb-m08-baseline-') as tmp:
        folder=Path(tmp);input_path=folder/'public.wav';snd.save(str(input_path),'WAV')
        # Each combination has a unique output directory, owned entirely by this capture.
        for i,(speed,ratio,hz) in enumerate([(1,1,0),(.8,1,0),(1,1.2,20),(1.5,.8,-10),(1.005,1.005,.005)]):
            dest=folder/f'transform{i}';dest.mkdir()
            parselmouth.praat.run("random_initializeWithSeedUnsafelyButPredictably (42)")
            try:
                path=namespace['process_single_file'](None,str(input_path),speed,ratio,hz,str(dest))
                data[f'transform_{i}']=parselmouth.Sound(path).values
            except parselmouth.PraatError as error:
                errors[f'transform_{i}']=str(error)
        data['pcm_input']=parselmouth.Sound(str(input_path)).values
        for a in ['full','order','reverse','constant']:
            for b in ['full','order','reverse','constant']:
                i=len(batch_cases);dest=folder/f'batch{i}';dest.mkdir()
                args=dict(t1=.1,t2=.5,f1_list=[180,120],f2_list=[140,200],knot_points=[dict(time=.3,freqs=[160,190])],start_mode=a,end_mode=b,knot_modes=['order'],offset_mode=False)
                parselmouth.praat.run("random_initializeWithSeedUnsafelyButPredictably (42)")
                batch.generate_batch_linear(snd,str(dest/'public.wav'),times,f0,0,.6,**args)
                files=sorted(dest.glob('*.wav'))
                for j,file in enumerate(files):data[f'batch_{i}_{j}']=parselmouth.Sound(str(file)).values
                batch_cases.append(dict(args=args,names=[f.name for f in files]))
        dest=folder/'offset';dest.mkdir()
        args=dict(t1=.1,t2=.5,f1_list=[-20],f2_list=[20],knot_points=[],start_mode='constant',end_mode='constant',knot_modes=[],offset_mode=True)
        batch.generate_batch_linear(snd,str(dest/'public.wav'),times,f0,0,.6,**args)
        file=next(dest.glob('*.wav'));data['offset']=parselmouth.Sound(str(file)).values
    hashes={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'core/manipulation/synthesis.py',SOURCE/'core/manipulation/batch_utils.py',SOURCE/'services/manipulation_service.py',SOURCE/'gui/widgets/pitch_manipulation_widget.py',SOURCE/'gui/dialogs/manipulation_dialogs.py']}
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'v2.npz',**data)
    (OUT/'v2.json').write_text(json.dumps(dict(sample_rate=sr,synth=cases,batch=batch_cases,errors=errors,source_hashes=hashes,parselmouth=parselmouth.__version__,numpy=np.__version__,array_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in data.items()}),indent=2),encoding='utf8')
    print(json.dumps(dict(arrays=len(data),values=sum(x.size for x in data.values()))))

if __name__=='__main__':capture()
