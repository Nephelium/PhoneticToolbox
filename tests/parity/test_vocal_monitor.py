"""M10-R17 waveform zoom: analytical PCM fixtures, no audio device."""
from types import SimpleNamespace
import numpy as np
import pytest
from phonetic_core.vocal_tract.monitor import OutputHistory, analyze_output
from ptb_desktop.vocal_tract.runtime import Runtime


def test_zoom_returns_exact_visible_pcm_including_history_lag():
    sr=48000
    samples=(.004*np.sin(2*np.pi*150*np.arange(sr)/sr)).astype(np.float32)
    original=samples.copy()
    result=analyze_output(samples,sr,seconds=1,end=7,view_seconds=.02,view_lag=.4)
    expected=samples[27840:28800]  # [6.58, 6.60) in the [6, 7) buffer
    assert result['waveform_mode']=='samples' and result['waveform']==[]
    assert result['waveform_start']==pytest.approx(6.58)
    assert result['waveform_duration']==.02 and result['sample_rate']==sr
    np.testing.assert_array_equal(result['waveform_samples'],expected)
    np.testing.assert_array_equal(samples,original)
    assert result['waveform_peak']==float(np.max(np.abs(expected)))


def test_overview_bins_only_visible_range_and_preserves_single_sample_peak():
    samples=np.zeros(48000,dtype=np.float32)
    samples[100]=.99  # outside requested view
    samples[40003]=-.002
    result=analyze_output(samples,48000,seconds=1,end=1,view_seconds=.5,waveform_width=640)
    assert len(result['waveform'])==640 and result['waveform_samples']==[]
    assert result['waveform_start']==.5 and result['waveform_duration']==.5
    assert result['waveform_peak']==pytest.approx(.002)
    assert min(p[0] for p in result['waveform'])==pytest.approx(-.002)
    assert max(p[1] for p in result['waveform'])==0


def test_wave_zoom_preserves_spectral_analysis_and_empty_startup():
    y=.2*np.sin(2*np.pi*1000*np.arange(48000)/48000)
    old=analyze_output(y,48000,seconds=1,end=1)
    new=analyze_output(y,48000,seconds=1,end=1,view_seconds=.001)
    assert old['spectrogram']==new['spectrogram']
    assert len(new['waveform_samples'])==48
    assert old['waveform_samples']==[] and len(old['waveform'])==700
    empty=analyze_output([],48000,view_seconds=.001)
    assert empty['waveform_samples']==[] and empty['waveform_peak']==0
    short=analyze_output(y[:240],48000,end=.005,view_seconds=.02)
    assert short['waveform_start']==0 and len(short['waveform_samples'])==240


@pytest.mark.parametrize('settings',[
    {'view_seconds':0},{'view_seconds':float('nan')},{'view_seconds':13},
    {'view_seconds':1,'view_lag':12},{'view_lag':-1},{'waveform_width':9000},
])
def test_monitor_rejects_invalid_view(settings):
    with pytest.raises(ValueError):analyze_output([],48000,**settings)


def test_runtime_route_keeps_wraparound_pcm_and_zoom_coordinates():
    history=OutputHistory(48000,seconds=12)
    t=np.arange(13*48000)/48000
    samples=(.007*np.sin(2*np.pi*200*t)).astype(np.float32)
    history.append(samples)
    runtime=Runtime.__new__(Runtime)
    runtime.engine=SimpleNamespace(sr=48000)
    runtime.live=SimpleNamespace(output_history=history)
    result=runtime.invoke('audio/monitor',{'seconds':2,'view_seconds':.01,'view_lag':1,'waveform_width':500})
    assert result['waveform_start']==pytest.approx(11.99)
    np.testing.assert_array_equal(result['waveform_samples'],samples[575520:576000])
