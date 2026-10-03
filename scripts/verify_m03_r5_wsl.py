"""Limited Linux pure-core checks; native REAPER/GUI not claimed."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import sawtooth
from phonetic_core.egg.f0 import praat_pitch,reaper_pitch
from phonetic_core.ports.acoustic import ReaperTrack

root=Path(__file__).resolve().parents[1]
checks=[];rate=16000;t=np.arange(rate)/rate
for frequency in (40,700):
    track=praat_pitch(.6*sawtooth(2*np.pi*frequency*t,width=.15),rate,pitch_floor=30.,pitch_ceiling=800.)
    assert np.isfinite(track.values).sum()>30
    assert abs(np.nanmedian(track.values)-frequency)/frequency<.03
    checks.append(dict(check='Praat actual tone',frequency_hz=frequency,median_hz=float(np.nanmedian(track.values))))
for count in (800,8000):
    assert not np.isfinite(praat_pitch(np.zeros(count),rate,pitch_floor=30.,pitch_ceiling=800.).values).any()
    checks.append('Praat no fabricated pitch on silent '+str(count)+' samples')
def port(audio,step,low,high,*,hilbert,no_highpass):
    assert (step,low,high,hilbert,no_highpass)==(.01,30.,800.,False,False)
    return ReaperTrack(np.array([.07,.08,.09]),np.array([40.,np.nan,700.]),'native_reaper')
track=reaper_pitch(np.sin(2*np.pi*120*t),rate,port)
np.testing.assert_array_equal(track.times,[.07,.08,.09]);assert np.isnan(track.values[1])
checks.append('Injected REAPER port: exact bounds/time/NaN, no native execution claim')
report=dict(success=True,platform='WSL NInfer',checks=checks,limits='Pure core only; injected port is not Linux native REAPER or GUI validation.')
(root/'output/m03-r5/wsl-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
