"""M06-R2 approved scientific revision: analytic source and task invariants."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from phonetic_core.synthesis.klatt.api import defaults,validate,generate,synthesize_with_info,extract,export_parameters,import_parameters
from phonetic_core.synthesis.klatt.tdklatt import KlattParam1980,klatt_make
from phonetic_core.synthesis.klatt.source_levels import source_gain,INTERNAL_RATE
from phonetic_core.models.audio import AudioInput


def config(duration=.3):
    c=defaults();c['duration']=duration
    for curve in c['curves'].values():curve['points'][-1][0]=duration
    return c


def source(av=60,ah=0,duration=.5):
    np.random.seed(20261002)
    s=klatt_make(KlattParam1980(FS=INTERNAL_RATE,DUR=duration,F0=120,AV=av,AH=ah,AF=0,AVS=0))
    s.run();return s


def rms(x):return np.sqrt(np.mean(x*x))


def test_control_gain_analytic_and_zero_sources():
    np.testing.assert_allclose(source_gain([0,20,40,60,80]),[0,.01,.1,1,10],rtol=1e-14)
    s=source(av=0,ah=0)
    assert not np.any(s.voice.av.output) and not np.any(s.voice.avs.output)
    assert not np.any(s.cascade.ah.output) and not np.any(s.parallel.af.output)
    assert not np.any(s.output)


def test_source_db_increment_and_fixed_reference():
    low=source(60);high=source(66)
    np.testing.assert_allclose(high.voice.av.output,low.voice.av.output*10**(.3),rtol=1e-13,atol=1e-15)
    # Steady last 20 complete periods, expected digital calibration RMS .01.
    period=round(INTERNAL_RATE/120)
    assert rms(low.voice.av.output[-20*period:])==pytest.approx(.01,rel=1e-10)
    whisper=source(0,60,duration=1.)
    assert not np.any(whisper.voice.switch.output[0])
    assert rms(whisper.cascade.ah.output)==pytest.approx(.01,rel=.035)


def test_noise_is_bounded_one_zero_fir_not_accumulator():
    s=source()
    expected=s.noise.noisegen.output.copy();expected[1:]+=s.noise.noisegen.output[:-1]
    np.testing.assert_array_equal(s.noise.lowpass.output,expected)
    np.testing.assert_allclose(s.noise.amp.output,expected*.001,rtol=1e-14,atol=1e-15)


def test_av_controls_final_amplitude_without_agc():
    c=config();c['fade_in']=c['fade_out']=0
    c['curves']['AV']['override']=50
    np.random.seed(1);a,info_a=synthesize_with_info(c)
    c['curves']['AV']['override']=56
    np.random.seed(1);b,info_b=synthesize_with_info(c)
    assert info_a['output_gain']==info_b['output_gain']==1
    assert 20*np.log10(rms(b)/rms(a))==pytest.approx(6,abs=.01)
    c['curves']['AV']['override']=0
    silent,info=synthesize_with_info(c)
    assert not np.any(silent)


def test_hnr_does_not_change_aspiration_control():
    c=config();c['curves']['AV']['override']=0;c['curves']['AH']['override']=40
    np.random.seed(2);a,_=synthesize_with_info(c)
    c['curves']['HNR']['override']=0
    np.random.seed(2);b,_=synthesize_with_info(c)
    np.testing.assert_array_equal(a,b)


def test_pcm_overload_reports_only_attenuation_and_preserves_silence():
    c=config();c['sequence']='a';c=generate(c)
    c['curves']['AV']['override']=80;c['curves']['AH']['override']=80
    np.random.seed(8);audio,info=synthesize_with_info(c)
    assert 0<info['output_gain']<1
    assert np.max(abs(audio))==pytest.approx(.95,abs=1e-12)
    assert info['fixed_output_gain']==64.


@pytest.mark.parametrize('vowel',['a','i','u'])
@pytest.mark.parametrize('f0',[60.,120.,300.])
def test_vowel_f0_finite_and_no_input_mutation(vowel,f0):
    c=config();c['sequence']=vowel;c=generate(c);c['curves']['F0']['override']=f0
    before=deepcopy(c);np.random.seed(3);audio,info=synthesize_with_info(c)
    assert c==before and len(audio)==4800 and np.isfinite(audio).all()
    assert 0<rms(audio)<.95 and np.max(abs(audio))<=.95000001
    assert info['internal_rate_hz']/2>6000


def test_vowel_generation_preserves_preset_sources_and_f0():
    c=config();c['sequence']='a i';c['curves']['AV']['override']=0;c['curves']['AH']['override']=40
    c['curves']['F0']['points']=[[0,300],[.3,350]];c['f0_transform']=dict(preset='假声',offset_hz=180.)
    generated=generate(c)
    assert generated['curves']['F0']==c['curves']['F0'] and generated['f0_transform']==c['f0_transform']
    assert not any(v for t,v in generated['curves']['AV']['points'])
    assert generated['curves']['AH']['override']==40


def test_new_serialization_and_legacy_rejection():
    c=config();c['f0_transform']=dict(preset='嘎裂',offset_hz=-30.)
    assert import_parameters(export_parameters(c))==c
    assert import_parameters(json.dumps(dict(schema_version='m06/1',config=c)))==c
    with pytest.raises(ValueError,match='m06_legacy'):validate(dict(c,schema_version='m06/1'))
    with pytest.raises(ValueError,match='m06_legacy'):import_parameters('Parameter,Time,Value,Global\nAV,0,200,True')
    c['curves']['AV']['points'][0][1]=200
    with pytest.raises(ValueError,match='m06_curve_value_range'):validate(c)


def test_extraction_initializes_source_envelope_and_clears_shift():
    fs=16000;t=np.arange(fs//2)/fs;audio=AudioInput((.1*np.sin(2*np.pi*150*t)).astype(np.float32),fs)
    c=config();c['f0_transform']=dict(preset='假声',offset_hz=50.)
    result=extract(c,audio)
    assert result['f0_transform']==dict(preset=None,offset_hz=0.)
    values=np.array(result['curves']['AV']['points'])[:,1]
    assert np.max(values)<=80 and np.median(values[values>0])==pytest.approx(60.,abs=.01)
    assert not any(v for _,v in result['curves']['AH']['points'])


def test_preset_file_has_no_f0_constants():
    from phonetic_core.synthesis.klatt import api
    presets=json.loads(Path(api.__file__).with_name('presets.json').read_text('utf8'))
    assert len(presets)==5 and all('F0' not in p for p in presets.values())
    assert presets['耳语']['AV']==0 and presets['耳语']['AH']>0
