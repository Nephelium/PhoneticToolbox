import numpy as np
import pytest
from ptb_api.spec2wav_models import Spec2WavConfig
from phonetic_core.spec2wav.adaptive import auto_gain, pulse_source, reconstruct_calibrated


def test_contract_limits_and_log_axis():
    assert Spec2WavConfig(n_iter=256).n_iter==256
    for config in [dict(n_iter=257),dict(n_iter=0),dict(frequency_axis='log',freq_start=0),
                   dict(mode='audio_draw',algorithm='neural_pipeline')]:
        with pytest.raises(ValueError):Spec2WavConfig(**config)
    assert Spec2WavConfig(frequency_axis='log',freq_start=50).frequency_axis=='log'
    old=Spec2WavConfig().snapshot()
    assert not {'algorithm','auto_parameters','frequency_axis'} & old.keys()


@pytest.mark.parametrize('level',[1e-7,.02,2.])
def test_gain_raises_quiet_and_limits_loud_without_clipping(level):
    x=np.sin(np.arange(2400)*2*np.pi/120)*level
    output,meta=auto_gain(x)
    assert np.sqrt(np.mean(output**2))==pytest.approx(.08)
    assert np.max(np.abs(output))<=.95
    np.testing.assert_allclose(output,x*meta['output_gain'])


def test_silence_and_transient_gain():
    y,_=auto_gain(np.zeros(100));assert np.all(y==0)
    x=np.zeros(10000);x[50]=1
    y,_=auto_gain(x);assert y[50]==pytest.approx(.95)


def test_parabolic_peak_impulse_has_expected_time_and_no_invented_pulses():
    y,n=pulse_source(np.array([[0.,1.,4.,1.,0.]]),np.arange(5)/100,1000,50)
    assert n==1 and np.argmax(y)==20 and y[20]==1
    flat,n=pulse_source(np.ones((3,5)),np.arange(5)/100,1000,50)
    assert n==0 and not flat.any()


@pytest.mark.parametrize('window,phase',[(14,'pulse'),(15,'random'),(16,'random')])
def test_phase_boundary_quarter_hop_and_duration(window,phase):
    gray=np.full((40,80),220,np.uint8);gray[8:12,::6]=20
    r=reconstruct_calibrated(gray,time_end=.123,freq_end=8000,win_length_ms=window,n_iter=2,target_sr=22050)
    m=r['metadata']
    assert m['phase_initialization']==phase and m['hop_length']*4==m['win_length']
    assert len(r['audio'])==round(.123*22050) and np.isfinite(r['audio']).all()
    assert np.abs(r['audio']).max()<=.95


def test_log_frequency_calibration_is_supported_and_distinct():
    gray=np.full((40,80),230,np.uint8);gray[20:24,:]=30
    args=dict(time_end=.12,freq_start=100,freq_end=8000,win_length_ms=20,n_iter=2,target_sr=24000)
    linear=reconstruct_calibrated(gray,**args)
    log=reconstruct_calibrated(gray,frequency_axis='log',**args)
    assert not np.allclose(linear['audio'],log['audio'])
    with pytest.raises(ValueError):reconstruct_calibrated(gray,frequency_axis='log')


def test_256_iterations_allowed_on_small_grid():
    gray=np.full((8,10),180,np.uint8)
    assert reconstruct_calibrated(gray,time_end=.05,freq_end=4000,win_length_ms=20,n_iter=256)['metadata']['n_iter']==256


def test_unpublished_pipeline_keeps_legacy_download_available(tmp_path,monkeypatch):
    from ptb_worker import vocoder_components as c
    monkeypatch.setenv('PTB_VOCODER_ROOT',str(tmp_path))
    monkeypatch.setattr(c.vocoder_catalog,'RELEASE',dict(id='new',pipeline=True,published=False,download_bytes=300))
    monkeypatch.setattr(c.vocoder_catalog,'PREVIOUS_RELEASE',dict(id='old',download_bytes=290,installed_bytes=840))
    result=c.status()
    assert result['available'] and not result['pipeline_available'] and not result['pipeline_ready']
    assert result['download_bytes']==290 and result['pipeline_download_bytes']==300


@pytest.mark.parametrize('window,choice',[(5,'random'),(20,'pulse')])
def test_explicit_initialization_overrides_threshold(window,choice):
    gray=np.full((40,80),220,np.uint8);gray[8:12,::6]=20
    result=reconstruct_calibrated(gray,time_end=.1,freq_end=5000,win_length_ms=window,n_iter=2,phase_initialization=choice,target_sr=0)
    assert result['metadata']['phase_initialization']==choice
    assert result['metadata']['threshold_basis']=='user_choice'
    assert result['sr']==10000 and len(result['audio'])==1000


@pytest.mark.parametrize('axis,lower',[('linear',1000),('log',100)])
def test_comparison_uses_requested_band_not_full_fft(axis,lower):
    from phonetic_core.spec2wav.adaptive import comparison_image
    sr=24000;t=np.arange(sr)/sr
    # A measured 2 kHz tone must appear at its input-axis coordinate.
    y=np.sin(2*np.pi*2000*t)*.1
    meta=dict(native_sr=sr,n_fft=1024,hop_length=120,win_length=480,schema_version='m09-calibrated/1')
    config=dict(time_start=0,time_end=1,freq_start=lower,freq_end=5000,frequency_axis=axis)
    image=comparison_image(y,sr,meta,config,(181,321))
    assert image.shape==(181,321)
    freqs=(np.geomspace if axis=='log' else np.linspace)(lower,5000,181)
    peak=np.argmin(image[:,30:-30].mean(1))
    assert abs(peak-(180-np.argmin(abs(freqs-2000))))<=2
