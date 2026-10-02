"""Read-only M06 gain audit. Fixed-seed experiments; never changes product code.

Run with the existing M09 Windows scientific environment and core PYTHONPATH.
Upstream source is inspected separately, never imported/executed by this script.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import parselmouth
from phonetic_core.synthesis.klatt.api import defaults
from phonetic_core.synthesis.klatt.engine import Engine
from phonetic_core.synthesis.klatt.tdklatt import KlattParam1980, klatt_make

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/validation/m06-r1/av-audit'
SEED = 20261002


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def stage(av, hnr=60., noise_db=0., mute_voice=False, mute_noise=False):
    # Same local core as production, observed before each normalized output.
    np.random.seed(SEED)
    params = KlattParam1980(DUR=.5, F0=120, AV=av, AH=max(0., 130-hnr),
                            Jitter=0, Shimmer=0, SHR=0, HNR=None, Slope=-10)
    s = klatt_make(params)
    s.voice.run()
    s.noise.run()
    voice_rms = rms(s.voice.switch.output[0])
    noise_rms = rms(s.noise.amp.output) * 10**(noise_db/20) * 10**(max(0.,130-hnr)/20)
    # Explicit zeroing only in this audit isolates the sources. In product,
    # AV=0/AH=0 use the generic amplifier and are NOT a source-off switch.
    if mute_voice:
        for b in s.cascade.ins[:1] + s.parallel.ins[:1]:
            b.input[:] = 0
    if mute_noise or noise_db:
        gain = 0 if mute_noise else 10**(noise_db/20)
        for b in s.cascade.ins[1:] + s.parallel.ins[1:]:
            b.input[:] *= gain
    s.cascade.run(); s.parallel.run(); s.radiation.run(); s.output_module.run()
    return dict(av=av,hnr=hnr,noise_db=noise_db,voice_rms=voice_rms,aspiration_rms=noise_rms,
                source_ratio_db=20*np.log10(max(voice_rms,1e-300)/max(noise_rms,1e-300))), s.output_module.output.copy()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    stages=[]
    for av in [0,60,120,190,200]:
        m,_=stage(av);stages.append(m)
    for av in [60,130,140]:
        m,_=stage(av,noise_db=-60);stages.append(m)
    final=[]
    for av in [0,60,120,190,200]:
        c=defaults();c['duration']=.5;c['fade_in']=0;c['fade_out']=0
        for v in c['curves'].values():v['points'][-1][0]=.5
        c['curves']['AV']['override']=av
        np.random.seed(SEED);audio=Engine(c).synthesize()
        snd=parselmouth.Sound(audio,sampling_frequency=c['sample_rate'])
        h=snd.to_harmonicity_cc(time_step=.01,minimum_pitch=75,silence_threshold=.1,periods_per_window=4.5).values.ravel()
        values=h[(h!=-200)&np.isfinite(h)]
        final.append(dict(av=av,peak=float(np.max(np.abs(audio))),rms=rms(audio),
                          measured_harmonicity_cc_db=float(np.mean(values)) if len(values) else None,
                          sha256=hashlib.sha256(audio.tobytes()).hexdigest()))
    # 0 dB still produces a nonzero periodic source, including AVS=0 leakage.
    zero,_=stage(0)
    assert zero['voice_rms']>0
    # Increasing AV by 10 dB almost multiplies the voiced branch by sqrt(10).
    assert abs(stages[4]['source_ratio_db']-stages[3]['source_ratio_db']-10)<.001
    assert all(np.isfinite(v['rms']) and abs(v['peak']-.95)<1e-12 for v in final)
    preset_metrics=[]
    presets=json.loads((ROOT/'frontend/src/modules/speech-synthesis/catalog.json').read_text(encoding='utf8'))['presets']
    for name in ['常态浊声','气声','耳语']:
        values=presets[name]
        m,_=stage(values['AV'],hnr=values['HNR'])
        preset_metrics.append(dict(name=name,parameters=values,source_ratio_db=m['source_ratio_db']))
    report=dict(seed=SEED,duration=.5,scope='fixed local defaults; digital source ratios and Praat autocorrelation harmonicity; no perceptual/physiological validation',
                stages=stages,final=final,presets=preset_metrics)
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
