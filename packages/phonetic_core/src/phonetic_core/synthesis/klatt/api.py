"""Validated, per-call M06 state. No device, persistence or process imports.

Numeric operations stay in engine.py. Validation and lossless serialization are
V3 boundary additions, listed independently in the migration report.
"""
from copy import deepcopy
import csv
import io
import json
import math
from .klatt_config import PARAM_DEFAULTS
from .input_parser import VOWEL_FORMANTS

VERSION = 'm06/2'
MAX_DURATION = 100.
MAX_SAMPLES = 4_800_000


def defaults():
    return dict(schema_version=VERSION, duration=2., sample_rate=16000, sequence='',
                fade_in=50, fade_out=100, smooth=5, f0_range=[50.,500.],
                curves={n:dict(points=[[0.,v[0]],[2.,v[0]]],override=None) for n,v in PARAM_DEFAULTS.items()},
                silence=[], boundaries=[], f0_transform=dict(preset=None, offset_hz=0.),
                f0_method='praat_cc')


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def validate(config):
    c=deepcopy(config)
    if isinstance(c,dict) and c.get('schema_version') == 'm06/1':
        raise ValueError('m06_legacy_parameters_need_revision')
    if isinstance(c,dict) and c.get('schema_version') == VERSION:
        c.setdefault('f0_method','praat_cc')
        # R4 Klatt snapshots remain readable. Retired engines must not silently
        # render through Klatt and produce a different scientific result.
        if 'render' in c:
            if not isinstance(c['render'],dict) or c['render'].get('method')!='klatt':
                raise ValueError('m06_removed_synthesis_method')
            c.pop('render')
        if c['f0_method']=='harvest':raise ValueError('m06_removed_f0_method')
    if not isinstance(c,dict) or set(c)!=set(defaults()) or c['schema_version']!=VERSION:
        raise ValueError('m06_invalid_config')
    if c['f0_method'] not in ('praat_cc','praat_ac','reaper'):
        raise ValueError('m06_invalid_f0_method')
    if not finite(c['duration']) or not .1<=c['duration']<=MAX_DURATION:raise ValueError('m06_duration_range')
    if type(c['sample_rate']) is not int or not 8000<=c['sample_rate']<=192000 or c['duration']*c['sample_rate']>MAX_SAMPLES:
        raise ValueError('m06_sample_budget')
    if not isinstance(c['sequence'],str) or len(c['sequence'])>2048:raise ValueError('m06_invalid_sequence')
    for n,lo,hi in [('fade_in',0,1000),('fade_out',0,1000),('smooth',1,50)]:
        if type(c[n]) is not int or not lo<=c[n]<=hi:raise ValueError('m06_invalid_'+n)
    r=c['f0_range']
    if not isinstance(r,list) or len(r)!=2 or not all(finite(x) for x in r) or not 1<=r[0]<r[1]<=3000:raise ValueError('m06_f0_range')
    transform=c['f0_transform']
    if (not isinstance(transform,dict) or set(transform)!={'preset','offset_hz'} or
            transform['preset'] not in (None,'假声','嘎裂') or not finite(transform['offset_hz']) or
            abs(transform['offset_hz'])>2999 or
            (transform['preset'] is None and transform['offset_hz']!=0) or
            (transform['preset']=='假声' and transform['offset_hz']<0) or
            (transform['preset']=='嘎裂' and transform['offset_hz']>0)):
        raise ValueError('m06_f0_transform')
    if not isinstance(c['curves'],dict) or set(c['curves'])!=set(PARAM_DEFAULTS):raise ValueError('m06_curve_keys')
    for name,curve in c['curves'].items():
        if not isinstance(curve,dict) or set(curve)!= {'points','override'}:raise ValueError('m06_invalid_curve')
        points=curve['points']
        if not isinstance(points,list) or not 1<=len(points)<=10001:raise ValueError('m06_curve_budget')
        if any(not isinstance(p,(list,tuple)) or len(p)!=2 or not all(finite(v) for v in p) or not 0<=p[0]<=c['duration'] for p in points):
            raise ValueError('m06_invalid_curve')
        if any(a[0]>b[0] for a,b in zip(points,points[1:])):raise ValueError('m06_curve_order')
        lo,hi=r if name=='F0' else PARAM_DEFAULTS[name][1:3]
        if any(not lo<=p[1]<=hi for p in points):raise ValueError('m06_curve_value_range')
        if name=='F0' and any(not 1<=p[1]-transform['offset_hz']<=3000 for p in points):raise ValueError('m06_f0_transform')
        if curve['override'] is not None:
            lo,hi=r if name=='F0' else PARAM_DEFAULTS[name][1:3]
            if not finite(curve['override']) or not lo<=curve['override']<=hi:raise ValueError('m06_override_range')
            if name=='F0' and not 1<=curve['override']-transform['offset_hz']<=3000:raise ValueError('m06_f0_transform')
    if not isinstance(c['silence'],list) or len(c['silence'])>2048:raise ValueError('m06_invalid_silence')
    for p in c['silence']:
        if not isinstance(p,(list,tuple)) or len(p)!=2 or not all(finite(v) for v in p) or not 0<=p[0]<p[1]<=c['duration']:raise ValueError('m06_invalid_silence')
    if not isinstance(c['boundaries'],list) or len(c['boundaries'])>2048 or any(not finite(v) or not 0<=v<=c['duration'] for v in c['boundaries']):raise ValueError('m06_invalid_boundaries')
    return c


def validate_sequence(text):
    # Keep original leading/trailing-space stripping and legal multiplicative grammar.
    if not text.strip():raise ValueError('m06_empty_sequence')
    previous=False
    for i,ch in enumerate(text):
        if ch.lower() in VOWEL_FORMANTS or ch==' ':previous=True
        elif ch in '+-*/' and previous:pass
        else:raise ValueError(f'm06_invalid_ipa:{i+1}:{ch}')
    from .input_parser import parse_vowel_sequence
    segments=parse_vowel_sequence(text.strip())
    if not any(s.type=='vowel' for s in segments) or not all(math.isfinite(s.duration_modifier) and s.duration_modifier>0 for s in segments):
        raise ValueError('m06_invalid_sequence')


def snapshot(engine, config):
    c=deepcopy(config)
    c['curves']={n:dict(points=[list(p) for p in v.points],override=v.global_override) for n,v in engine.params.items()}
    c['silence']=[list(p) for p in engine.silence_intervals];c['boundaries']=list(engine.vowel_boundaries)
    return validate(c)


def generate(config):
    from .engine import Engine
    c=validate(config);validate_sequence(c['sequence']);engine=Engine(c);engine.generate_vowels()
    return snapshot(engine,c)


def synthesize(config, *, cancelled=lambda:False):
    return synthesize_with_info(config, cancelled=cancelled)[0]


def synthesize_with_info(config, *, cancelled=lambda:False):
    import numpy as np
    from .engine import Engine
    c=validate(config)
    if cancelled():raise InterruptedError('m06_cancelled')
    engine=Engine(c);audio=engine.synthesize()
    if cancelled():raise InterruptedError('m06_cancelled')
    if audio.size!=round(c['duration']*c['sample_rate']) or not np.isfinite(audio).all():raise ValueError('m06_invalid_output')
    return audio, dict(computation_revision='klatt/2', internal_rate_hz=20000,
                       source_reference_db=60., source_reference_rms=.01,
                       fixed_output_gain=64., output_gain=engine.output_gain)


def extract(config, audio, *, cancelled=lambda:False, reaper=None, diagnostics=None):
    import numpy as np
    from .engine import Engine
    c=validate(config)
    if cancelled():raise InterruptedError('m06_cancelled')
    # V2 soundfile float32 decode then channel mean, promoted to float64.
    channels=audio.normalized_channels().astype(np.float32)
    mono=(np.mean(channels,axis=1) if channels.ndim>1 else channels).astype(float)
    c['sample_rate']=audio.sample_rate_hz;c['duration']=len(mono)/c['sample_rate']
    if not .1<=c['duration']<=MAX_DURATION or mono.size>MAX_SAMPLES:raise ValueError('m06_sample_budget')
    # Loading V2 kept old curves until extraction replaced them. All 23 tracks
    # are set by extract; initialize a valid duration-scaled configuration first.
    for curve in c['curves'].values():curve['points']=[[0.,curve['points'][0][1]],[c['duration'],curve['points'][-1][1]]]
    c['silence']=[];c['boundaries']=[]
    c['f0_transform']=dict(preset=None,offset_hz=0.)
    engine=Engine(c);engine.extract(audio,mono,reaper=reaper)
    if diagnostics is not None:
        diagnostics.update(engine.extraction_info)
    if cancelled():raise InterruptedError('m06_cancelled')
    return snapshot(engine,c)


def export_parameters(config):
    c=validate(config);stream=io.StringIO(newline='');writer=csv.writer(stream)
    writer.writerow(['Parameter','Time','Value','Global'])
    writer.writerow(['__DURATION__',c['duration'],0,False]);writer.writerow(['__VOWEL_INPUT__',0,c['sequence'],False])
    # Legacy readers ignore this metadata row; V3 retains ALL points beneath an
    # active scalar override, sample rate, fades, silence, range and smoothness.
    writer.writerow(['__PTB_CONFIG__',0,json.dumps(c,ensure_ascii=False,allow_nan=False,separators=(',',':')),False])
    for n,curve in c['curves'].items():
        if curve['override'] is not None:writer.writerow([n,0,curve['override'],True])
        else:
            for t,v in curve['points']:writer.writerow([n,t,v,False])
    return stream.getvalue()


def import_parameters(text):
    if not isinstance(text,str) or len(text.encode('utf8'))>8_000_000:raise ValueError('m06_parameter_budget')
    if text.lstrip().startswith('{'):
        value=json.loads(text)
        if isinstance(value,dict) and value.get('schema_version') in ('m06/1',VERSION) and isinstance(value.get('config'),dict):value=value['config']
        return validate(value)
    reader=csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
    if reader.fieldnames!=['Parameter','Time','Value','Global']:raise ValueError('m06_csv_header')
    rows=list(reader)
    full=[r for r in rows if r['Parameter']=='__PTB_CONFIG__']
    if len(full)>1:raise ValueError('m06_duplicate_snapshot')
    if full:return validate(json.loads(full[0]['Value']))
    raise ValueError('m06_legacy_parameters_need_revision')
