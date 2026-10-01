"""P16: independent time-grid, effective-parameter and failure regressions."""
from types import SimpleNamespace
import numpy as np
import pytest
from phonetic_core.acoustic.common import segment_for_frame
from phonetic_core.acoustic.energy import compute_energy, compute_rms
from phonetic_core.acoustic.formants_praat import compute_praat_formants
from phonetic_core.models.audio import AudioInput
from phonetic_core.models.acoustic import AcousticConfig
from phonetic_core.services import acoustic
from phonetic_core.ports.errors import BackendAborted


@pytest.mark.parametrize('fs,hop,k', [(44100,5.,2000),(22050,5.,2000),(48000,5.,2000),(44100,2.3,4000)])
def test_frame_center_does_not_accumulate_rounding(fs,hop,k):
    y=np.arange(fs*11,dtype=float)
    segment=segment_for_frame(y,fs,hop,k,3,200.)
    # Half-open window midpoint; allowed rounding is one sample, not one per frame.
    assert abs((segment[0]+segment[-1]+1)/2 - k*fs*hop/1000) <= 1


def test_energy_and_rms_sample_the_declared_time():
    fs=44100; k=2000; samples=np.zeros(fs*11)
    samples[fs*10-20:fs*10+20]=.5
    f0=np.zeros(k+1)
    assert compute_rms(samples,fs,5,f0,window_ms=2)[k] > .3
    assert compute_energy(samples,fs,5,f0,energy_window_ms=2)[k] > 80


@pytest.mark.parametrize('count',[3,4,5,7])
def test_burg_receives_the_requested_count(count):
    observed=[]
    track=SimpleNamespace(get_value_at_time=lambda n,t:500.*n,get_bandwidth_at_time=lambda n,t:50.)
    def burg(**kwargs):observed.append(kwargs);return track
    audio=SimpleNamespace(praat_sound=lambda:SimpleNamespace(duration=.03,to_formant_burg=burg))
    compute_praat_formants(audio,5.,num_formants=count)
    assert observed[0]['max_number_of_formants']==count


@pytest.mark.parametrize('name',[
    'compute_energy','compute_praat_formants','compute_praat_f0_track',
    'compute_spectral_features_batch','compute_H1H2_H2H4_corrected',
    'compute_cpp','compute_hnr','compute_shr','compute_spectral_slope',
    'compute_soe','compute_jitter_shimmer',
])
def test_stage_exception_cannot_become_a_successful_result(monkeypatch,name):
    audio=AudioInput(np.zeros((8000,1)),16000)
    def fail(*args,**kwargs):raise RuntimeError('private-path-must-not-escape')
    monkeypatch.setattr(acoustic,name,fail)
    with pytest.raises(Exception,match='analysis_.*_failed') as error:
        acoustic.analyze_audio(audio,AcousticConfig(use_reaper=False,smooth_win_size=0))
    assert 'private-path' not in str(error.value)


def test_cancellation_in_a_stage_is_not_swallowed(monkeypatch):
    def abort(*args,**kwargs):raise BackendAborted('cancelled')
    monkeypatch.setattr(acoustic,'compute_praat_formants',abort)
    with pytest.raises(BackendAborted):
        acoustic.analyze_audio(AudioInput(np.zeros((8000,1)),16000),AcousticConfig(use_reaper=False))


def test_silence_is_a_valid_missing_measurement():
    result=acoustic.analyze_audio(AudioInput(np.zeros((8000,1)),16000),AcousticConfig(use_reaper=False))
    assert len(result.time_axis)>0
    assert np.isnan(result.f0_praat).all()
