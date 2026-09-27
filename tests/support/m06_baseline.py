"""Independent V2 capture. No phonetic_core imports; GUI calls are inert only.

Original numeric methods are compiled unchanged from the neighbouring V2 widget.
No widgets, device streams, user files or V2 bytecode are created.
"""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
from typing import Optional
from math import gcd
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent/'PhoneticToolbox_v2'/'phonetic_toolbox'
OUT = ROOT/'tests/fixtures/m06'
METHODS = ['_get_neighbor_vowel_formants', 'generate_vowels', '_resize_to',
           '_slice_array_by_time', '_merge_silence_intervals', '_apply_fade',
           '_compute_rms', '_match_rms', '_synthesize_single_segment', 'synthesize',
           '_fit_track_to_len', '_sanitize_track_for_param', '_apply_track_to_curve',
           '_extract_loaded_audio_params']


class SilentUI:
    def __init__(self, value=None): self.v=value
    def text(self): return str(self.v)
    def value(self): return self.v
    def __getattr__(self, key): return lambda *a, **k: None


class Errors:
    @staticmethod
    def warning(*args): raise ValueError(str(args[-1]))
    critical = warning


def original():
    sys.dont_write_bytecode = True
    # Register namespace packages without executing V2 GUI/application __init__.
    for name, path in [('phonetic_toolbox',SOURCE),('phonetic_toolbox.core',SOURCE/'core'),
                       ('phonetic_toolbox.models',SOURCE/'models'),
                       ('phonetic_toolbox.core.acoustic',SOURCE/'core/acoustic'),
                       ('phonetic_toolbox.core.synthesis',SOURCE/'core/synthesis')]:
        mod=types.ModuleType(name);mod.__path__=[str(path)];sys.modules[name]=mod
    # tdklatt imports the device module, but capture never calls play().
    sys.modules['sounddevice']=types.ModuleType('sounddevice')
    klatt=importlib.import_module('phonetic_toolbox.core.synthesis.klatt')
    namespace=dict(np=np,Optional=Optional,gcd=gcd,Path=Path,
                   uniform_filter1d=uniform_filter1d,resample_poly=resample_poly,
                   QApplication=SilentUI(),QMessageBox=Errors)
    namespace.update({key:getattr(klatt,key) for key in klatt.__all__})
    for name, fn in [('f0_praat','compute_praat_f0'),('formants_praat','compute_praat_formants'),
                     ('energy','compute_energy'),('hnr','compute_hnr'),('shr','compute_shr'),
                     ('voicing','compute_voiced_mask'),('jitter_shimmer','compute_jitter_shimmer'),
                     ('spectral_slope','compute_spectral_slope'),('spectral_batch','compute_spectral_features_batch')]:
        namespace[fn]=getattr(importlib.import_module('phonetic_toolbox.core.acoustic.'+name),fn)
    tree=ast.parse((SOURCE/'gui/widgets/speech_synthesis_widget.py').read_text(encoding='utf8'))
    curve=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ParameterCurve')
    widget=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SpeechSynthesisWidget')
    selected=[n for n in widget.body if isinstance(n,ast.FunctionDef) and n.name in METHODS]
    cls=ast.ClassDef(name='Original',bases=[],keywords=[],body=selected,decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[curve,cls],type_ignores=[])),str(SOURCE/'gui/widgets/speech_synthesis_widget.py'),'exec'),namespace)
    return namespace


def instance(ns, duration=0.2, sequence='', fs=16000):
    obj=ns['Original']();obj.duration=duration;obj.fs=fs
    obj.params={n:ns['ParameterCurve'](n,*values[:3],duration) for n,values in ns['PARAM_DEFAULTS'].items()}
    obj.silence_intervals=[];obj.vowel_boundaries=[]
    obj.vowel_input=SilentUI(sequence);obj.smooth_slider=SilentUI(5)
    obj.fade_in_input=SilentUI(50);obj.fade_out_input=SilentUI(100)
    obj.workspace=SilentUI();obj.status_label=SilentUI();obj.audio_panel=SilentUI()
    obj._sync_x_range=lambda *a:None;obj.f0_min_hz=50.;obj.f0_max_hz=500.
    return obj


def capture():
    ns=original();arrays={};cases=[]
    settings=[(.1,'',16000,{}),(.6,'a-i/- ///u+/e++o/-',16000,{}),
              (.3,'ɨʉɯəɐɻ',44100,{}),(.25,'a',16000,{'Jitter':3.,'Shimmer':.03,'SHR':.8}),
              (.2,'i u',16000,{'HNR':20.,'H1H2':10.}),(.2,'u',48000,{'F0':300.})]
    for i,(duration,text,fs,overrides) in enumerate(settings):
        obj=instance(ns,duration,text,fs)
        if text: obj.generate_vowels()
        for n,v in overrides.items():obj.params[n].global_override=v
        curves={n:dict(points=c.points,override=c.global_override) for n,c in obj.params.items()}
        for n,c in obj.params.items(): arrays[f'{i}_{n}']=c.get_array(duration,fs)
        np.random.seed(420+i);obj.synthesize();arrays[f'{i}_audio']=obj.synthesized_audio
        cases.append(dict(duration=duration,sequence=text,sample_rate=fs,curves=curves,
                          silence=obj.silence_intervals,boundaries=obj.vowel_boundaries,seed=420+i))
    OUT.mkdir(parents=True,exist_ok=True)
    # Public synthetic source for original V2 path-based extraction.
    from scipy.io import wavfile
    t=np.arange(9600)/16000
    source=(.2*np.sin(2*np.pi*150*t)+.08*np.sin(2*np.pi*300*t)).astype(np.float32)
    wavfile.write(OUT/'source.wav',16000,source)
    obj=instance(ns,.6);obj.loaded_audio=source.astype(float);obj.loaded_audio_path=str(OUT/'source.wav')
    obj._extract_loaded_audio_params()
    for n,c in obj.params.items():arrays['extracted_'+n]=np.asarray(c.points)
    files=[SOURCE/'gui/widgets/speech_synthesis_widget.py',*sorted((SOURCE/'core/synthesis/klatt').glob('*.py')),
           *sorted((SOURCE/'core/acoustic').glob('*.py')),SOURCE.parent/'Phonetic_Export/index.html']
    hashes={str(p.relative_to(SOURCE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    np.savez_compressed(OUT/'v2.npz',**arrays)
    import scipy,parselmouth
    (OUT/'v2.json').write_text(json.dumps(dict(cases=cases,source_hashes=hashes,
        numpy=np.__version__,scipy=scipy.__version__,parselmouth=parselmouth.__version__,
        arrays={k:dict(shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}),ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(arrays=len(arrays),values=sum(v.size for v in arrays.values()))))


if __name__=='__main__':capture()
