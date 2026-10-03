"""R4 analytic signals and independent upstream calls, not naturalness claims."""
from copy import deepcopy
import numpy as np
import pytest
import pyworld
from phonetic_core.models.audio import AudioInput
from phonetic_core.acoustic.f0_praat import compute_praat_f0_track
from phonetic_core.synthesis.klatt.api import defaults,validate,export_parameters,import_parameters
from phonetic_core.synthesis.resynthesis import extract_natural,resynthesize


def source(fs=16000,kind='harmonic'):
    t=np.arange(fs)/fs
    x=.08*sum(np.sin(2*np.pi*150*k*t)/k for k in range(1,40))
    if kind=='mixed':
        x[4000:8000]=np.random.default_rng(12).normal(0,.03,4000)
        x[10000:14000]=0
    if kind=='silent':x[:]=0
    return AudioInput(x,fs)


def config(method='world',algorithm='praat_cc'):
    c=defaults();c['render']['method']=method;c['f0_method']=algorithm
    return extract_natural(c,source())[0]


def test_world_matches_upstream_analysis_and_synthesis():
    a=source();before=a.samples.copy();c=config(algorithm='harvest')
    output,info,arrays=resynthesize(c,a)
    f0,t=pyworld.harvest(a.scientific_mono(),16000,f0_floor=50.,f0_ceil=500.,frame_period=10.)
    size=pyworld.get_cheaptrick_fft_size(16000,f0_floor=50.)
    sp=pyworld.cheaptrick(a.scientific_mono(),f0,t,16000,f0_floor=50.,fft_size=size)
    ap=pyworld.d4c(a.scientific_mono(),f0,t,16000,fft_size=size)
    np.testing.assert_array_equal(arrays['source_f0_hz'],f0)
    np.testing.assert_array_equal(arrays['spectral_envelope_power'],sp)
    np.testing.assert_array_equal(arrays['aperiodicity_amplitude_ratio'],ap)
    ap=np.clip(ap,1e-12,1-1e-12);ap[f0<=0]=1-1e-12
    expected=pyworld.synthesize(f0,sp,ap,16000,frame_period=10.)[:16000]
    expected*=min(1.,.99/np.max(np.abs(expected)))
    np.testing.assert_array_equal(output,expected)
    np.testing.assert_array_equal(a.samples,before)
    assert info['pyworld_version']=='0.3.5' and info['actual_f0_backend']=='world_harvest'


@pytest.mark.parametrize('method',['world','psola'])
@pytest.mark.parametrize('duration',[.75,1.,1.5])
def test_edited_pitch_and_output_time(method,duration):
    c=config(method);c['duration']=duration
    for curve in c['curves'].values():curve['points']=[[t*duration,v] for t,v in curve['points']]
    c['curves']['F0']['override']=220.;c['render']['pitch']='curve'
    y,info,arrays=resynthesize(c,source())
    assert len(y)==round(16000*duration)
    measured=compute_praat_f0_track(AudioInput(y,16000),10,50,500)
    assert np.nanmedian(measured.values)==pytest.approx(220,abs=2)
    assert arrays['target_f0_hz'][arrays['target_f0_hz']>0].tolist()==[220.]*(arrays['target_f0_hz']>0).sum()
    assert info['output_gain']<=1


@pytest.mark.parametrize('method',['world','psola'])
def test_noise_retained_and_silence_not_voiced(method):
    c=config(method);a=source(kind='mixed');c['curves']['F0']['override']=220;c['render']['pitch']='curve'
    y,info,arrays=resynthesize(c,a)
    assert np.sqrt(np.mean(y[5000:7000]**2))>.005
    assert np.sqrt(np.mean(y[11500:12500]**2))<1e-5
    assert not arrays['source_f0_hz'][70:80].any()
    assert not arrays['target_f0_hz'][70:80].any()


@pytest.mark.parametrize('method',['world','psola'])
def test_all_silence_and_source_input_nonmutation(method):
    c=config(method);a=source(kind='silent');before=a.samples.copy()
    y,info,_=resynthesize(c,a)
    assert np.max(np.abs(y))<1e-5
    assert info['voiced_frames']==0
    np.testing.assert_array_equal(a.samples,before)


def test_world_spectrum_and_noise_controls_change_output_without_mutating_input():
    c=config();a=source();plain,_,arrays=resynthesize(c,a)
    c['render']['spectral_ratio']=1.2;c['render']['aperiodicity_ratio']=.5
    changed,info,other=resynthesize(c,a)
    assert np.max(np.abs(changed-plain))>1e-3
    assert info['spectral_ratio']==1.2 and info['aperiodicity_ratio']==.5
    np.testing.assert_array_equal(arrays['spectral_envelope_power'],other['spectral_envelope_power'])


def test_config_roundtrip_and_budget_errors():
    old=defaults();del old['render'];before=deepcopy(old)
    assert validate(old)['render']['method']=='klatt' and old==before
    for method in ['world','psola']:
        c=config(method);assert import_parameters(export_parameters(c))==c
    c=config();c['duration']=3.
    with pytest.raises(ValueError,match='m06_resynthesis_duration'):resynthesize(c,source())
    c=config();c['f0_range']=[20,500]
    with pytest.raises(ValueError,match='m06_world_input_range'):resynthesize(c,source())
    c=config();c['curves']['F0']['override']=1200;c['f0_range']=[50,1500];c['render']['pitch']='curve'
    with pytest.raises(ValueError,match='m06_world_input_range'):resynthesize(c,source())
    c=config();c['render']['source_sha256']='invalid'
    with pytest.raises(ValueError,match='m06_invalid_render'):validate(c)


def test_psola_unmodified_matches_direct_praat():
    from parselmouth import Sound
    from parselmouth.praat import call,run
    c=config('psola');a=source()
    run('random_initializeWithSeedUnsafelyButPredictably (12345)')
    y,info,arrays=resynthesize(c,a)
    run('random_initializeWithSeedUnsafelyButPredictably (12345)')
    manipulation=call(Sound(a.scientific_mono(),16000),'To Manipulation',.01,50.,500.)
    tier=call('Create PitchTier','comparison',0.,1.)
    for t,f in zip(arrays['source_times_s'],arrays['source_f0_hz']):
        if f>0:call(tier,'Add point',t,f)
    call([manipulation,tier],'Replace pitch tier')
    duration=call('Create DurationTier','duration',0.,1.);call(duration,'Add point',0.,1.)
    call([manipulation,duration],'Replace duration tier')
    expected=call(manipulation,'Get resynthesis (overlap-add)').values[0]
    np.testing.assert_array_equal(y,expected)
    assert info['pulse_backend']=='Praat To Manipulation'


def test_world_native_rate_and_matrix_limits():
    with pytest.raises(ValueError,match='m06_world_input_range'):resynthesize(config(),source(8000))
    fs=48000;t=np.arange(5*fs)/fs;a=AudioInput(.1*np.sin(2*np.pi*150*t),fs)
    c=config();c['duration']=10
    with pytest.raises(ValueError,match='m06_world_matrix_budget'):resynthesize(c,a)


def test_psola_refuses_unvoiced_pitch_edit_instead_of_ignoring_it():
    c=config('psola');c['render']['pitch']='curve';c['curves']['F0']['override']=220
    with pytest.raises(ValueError,match='m06_psola_no_voiced'):resynthesize(c,source(kind='silent'))


def test_optional_world_failure_is_explicit(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'pyworld',None)
    with pytest.raises(ValueError,match='m06_world_unavailable'):resynthesize(config(),source())
