"""Fixed-seed measured M06/2 preset matrix and shareable synthetic WAVs."""
from pathlib import Path
import json
import hashlib
import numpy as np
import soundfile as sf
import parselmouth
from phonetic_core.synthesis.klatt.api import defaults,generate,synthesize_with_info

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/validation/m06-r2/science'
PRESETS=json.loads((ROOT/'packages/phonetic_core/src/phonetic_core/synthesis/klatt/presets.json').read_text('utf8'))


def make(name,vowel,f0,duration):
    c=defaults();c['duration']=duration;c['sequence']=vowel
    for curve in c['curves'].values():curve['points'][-1][0]=duration
    c=generate(c)
    for key,value in PRESETS[name].items():c['curves'][key]['override']=value
    c['curves']['F0']['points']=[[0.,f0],[duration,f0]]
    return c


def measure(audio,fs):
    sound=parselmouth.Sound(audio,sampling_frequency=fs)
    f0=sound.to_pitch_cc(time_step=.01,pitch_floor=40,pitch_ceiling=700).selected_array['frequency']
    harmonicity=sound.to_harmonicity_cc(time_step=.01,minimum_pitch=40,silence_threshold=.1,periods_per_window=4.5).values.ravel()
    defined=harmonicity[np.isfinite(harmonicity)&(harmonicity!=-200)]
    return dict(rms=float(np.sqrt(np.mean(audio**2))),peak=float(np.max(abs(audio))),
                measured_f0_median=float(np.median(f0[f0>0])) if np.any(f0>0) else None,
                measured_voiced_fraction=float(np.mean(f0>0)),
                measured_harmonicity_cc=float(np.mean(defined)) if len(defined) else None)


def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for name in PRESETS:
        for vowel in ['a','i','u']:
            for f0 in ([50.,70.,110.] if name=='嘎裂' else [250.,300.,450.] if name=='假声' else [80.,120.,250.]):
                for duration in [.3,.8]:
                    c=make(name,vowel,f0,duration);np.random.seed(20261002)
                    audio,info=synthesize_with_info(c)
                    assert len(audio)==round(duration*16000) and np.isfinite(audio).all()
                    assert np.max(abs(audio))<=.95000001 and np.any(audio)
                    rows.append(dict(preset=name,vowel=vowel,f0=f0,duration=duration,parameters=PRESETS[name],diagnostics=info,**measure(audio,16000)))
    for name in PRESETS:
        c=make(name,'a-i-u',120.,2.)
        offset=172.5 if name=='假声' else -57.5 if name=='嘎裂' else 0.
        c['curves']['F0']['points']=[[0.,110.+offset],[1.,140.+offset],[2.,120.+offset]]
        c['f0_transform']=dict(preset=name if offset else None,offset_hz=offset)
        c['f0_range']=[20.,700.]
        np.random.seed(20261002);audio,info=synthesize_with_info(c)
        sf.write(OUT/(name+'.wav'),audio,16000,subtype='PCM_16')
        (OUT/(name+'.json')).write_text(json.dumps(dict(config=c,diagnostics=info),ensure_ascii=False,indent=2),encoding='utf8')
    result=dict(seed=20261002,scope='90 digital signal measurements; no listening/physiological validation',cases=rows)
    (OUT/'matrix.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    for name in PRESETS:
        group=[r for r in rows if r['preset']==name]
        print(name,'cases',len(group),'RMS',min(r['rms'] for r in group),max(r['rms'] for r in group),'limited',sum(r['diagnostics']['output_gain']<1 for r in group))
    print(OUT)


if __name__=='__main__':main()
