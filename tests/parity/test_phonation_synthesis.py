"""Expected values come from independent pre-migration V2 capture."""
import json
from pathlib import Path
import numpy as np
import pytest
from phonetic_core.manipulation import m07_api as core, m07_legacy as legacy
from phonetic_core.manipulation.m07_models import PhonationAnalysisConfig,PhonationGenerationConfig
from ptb_worker.m07_science import estimator

ROOT=Path(__file__).resolve().parents[1]/'fixtures/m07'
META=json.loads((ROOT/'v2.json').read_text(encoding='utf8'))
DATA=np.load(ROOT/'v2.npz',allow_pickle=False)

@pytest.fixture(scope='module')
def analyses():
    return [core.analyze(DATA[f'input{i}'].astype(float)/32768,c['fs'],PhonationAnalysisConfig(**c['config']),estimator()) for i,c in enumerate(META['cases'])]

@pytest.mark.parametrize('i',range(4))
def test_exact_analysis(analyses,i):
    item=analyses[i]
    for key in ('signal','residual','lpc_coefficients','f0_hz','pulses'):np.testing.assert_array_equal(getattr(item,key),DATA[f'{i}_{key}'])
    np.testing.assert_array_equal([item.start_sample,item.end_sample],DATA[f'{i}_bounds'])
    c=META['cases'][i];np.testing.assert_array_equal(core.resample_audio(DATA[f'input{i}'].astype(float)/32768,c['fs'],11025),DATA[f'{i}_resampled'])
    np.testing.assert_array_equal(estimator()(item.signal,11025,item.config),DATA[f'{i}_initial_f0'])

@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('kind',[1,2,3])
@pytest.mark.parametrize('energy',[False,True])
@pytest.mark.parametrize('normalize',[False,True])
def test_exact_generation(analyses,reverse,kind,energy,normalize):
    from scipy.io import wavfile
    import io
    label=f'g_{int(reverse)}_{kind}_{int(energy)}_{int(normalize)}'
    gen=PhonationGenerationConfig(3,energy,normalize,.83)
    result=core.generate(*analyses[:2],kind,gen,reverse)
    np.testing.assert_array_equal(result.audio_steps,DATA[label])
    pair=analyses[:2][::-1] if reverse else analyses[:2]
    np.testing.assert_array_equal(legacy.make_residual_continuum(*pair,kind,3,energy),DATA[label+'_residual'])
    for i,samples in enumerate([*result.audio_steps.T,result.audio_steps.T.reshape(-1)]):
        stream=io.BytesIO();wavfile.write(stream,11025,core.pcm16(samples))
        assert stream.getvalue()==DATA[label+f'_wav{i}'].tobytes()

@pytest.mark.parametrize('mode',['normalize','onset'])
def test_exact_controls_and_lpc_reuse(analyses,mode):
    controls=core.controls(*analyses[:2],21,mode)
    np.testing.assert_array_equal(controls['axis'],DATA[f'{mode}_axis'])
    pair=core.apply_controls(*analyses[:2],controls,mode)
    for i,item in enumerate(pair):
        np.testing.assert_array_equal(controls[['source','target'][i]],DATA[f'{mode}_{i}_controls'])
        np.testing.assert_array_equal(item.f0_hz,DATA[f'{mode}_{i}_edit'])
        assert item.pulses is analyses[i].pulses and item.lpc_coefficients is analyses[i].lpc_coefficients and item.residual is analyses[i].residual

@pytest.mark.parametrize('n',[1,63,128,129,160,1000])
def test_exact_tail(n):
    audio=np.arange(n,dtype=float)/n
    np.testing.assert_array_equal(legacy.enframe(audio,128,32),DATA[f'frames{n}'])
    coeff,res=legacy.build_lpc_residual(audio,0,n-1,128,32,20)
    np.testing.assert_array_equal(coeff,DATA[f'coeff{n}']);np.testing.assert_array_equal(res,DATA[f'tail{n}'])

@pytest.mark.parametrize('audio',[np.zeros(8000),np.ones(8)*.1,np.ones(4000)*.1,np.array([np.nan])])
def test_invalid_input(audio):
    with pytest.raises(ValueError):core.analyze(audio,8000,PhonationAnalysisConfig(),estimator())

def test_no_nan_quantization():
    with pytest.raises(ValueError,match='nonfinite'):core.pcm16([np.nan])

def test_backend_missing():
    with pytest.raises(ValueError,match='reaper_unavailable'):estimator()(np.ones(4000),11025,PhonationAnalysisConfig(f0_backend='reaper'))


def test_insufficient_pulses_and_input_limit():
    with pytest.raises(ValueError,match='脉冲不足'):
        legacy.detect_pulses(np.zeros(300),np.full(30,120.),11025,0,299)
    with pytest.raises(ValueError,match='input_budget'):
        core.analyze(np.zeros(480001),48000,PhonationAnalysisConfig(),estimator())


def test_control_time_and_nonfinite_rejection(analyses):
    controls=core.controls(*analyses[:2]);controls['axis'][2]=controls['axis'][1]
    with pytest.raises(ValueError,match='time_order'):core.apply_controls(*analyses[:2],controls,'normalize')
    controls=core.controls(*analyses[:2]);controls['source'][0]=float('inf')
    with pytest.raises(ValueError,match='nonfinite'):core.apply_controls(*analyses[:2],controls,'normalize')
