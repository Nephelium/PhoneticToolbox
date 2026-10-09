from dataclasses import replace
import numpy as np
import pytest
from phonetic_core.models.acoustic import AcousticConfig
from phonetic_core.models.audio import AudioInput
from phonetic_core.services.acoustic import analyze_audio
from phonetic_core.services.acoustic_stream import validate_source,layout,gap_interpolate,iter_parameters

def test_exact_30_minute_boundary():
    validate_source(1800*48000,48000,2)
    with pytest.raises(ValueError,match='duration_limit'):validate_source(1800*48000+1,48000,2)

def test_global_frame_ownership_has_no_duplicates_or_missing_frames():
    c=AcousticConfig(frameshift_ms=5)
    parts=list(layout(1800*44100,44100,c))
    indexes=np.concatenate([np.arange(a,b) for a,b,*_ in parts])
    np.testing.assert_array_equal(indexes,np.arange(360000))
    assert max(e-s for _,_,_,s,e in parts)<44100*24

def test_interpolation_keeps_nan_gaps_and_does_not_cross_removed_cycles():
    t=np.array([0.,.01,.02,.5,.51]);v=np.array([1.,np.nan,3.,4.,5.])
    out=gap_interpolate(t,v,[0.,.005,.015,.02,.1,.5,.505,.6],.04)
    np.testing.assert_allclose(out,[1,np.nan,np.nan,3,np.nan,4,4.5,np.nan],equal_nan=True)

def test_selective_execution_preserves_selected_numerics(monkeypatch):
    c=AcousticConfig(use_reaper=False,selected_parameter_keys=('pF0','Intensity'),smooth_win_size=1)
    y=.3*np.sin(2*np.pi*180*np.arange(16000)/16000)
    # Unselected heavy stages must not run.
    import phonetic_core.services.acoustic as module
    for name in ('compute_cpp','compute_hnr','compute_jitter_shimmer','compute_spectral_features_batch'):
        monkeypatch.setattr(module,name,lambda *a,**k: (_ for _ in ()).throw(AssertionError('unselected stage')))
    r=analyze_audio(AudioInput(y,16000),c,selective=True)
    assert len(r.f0_praat)>100
    assert np.nanmedian(r.f0_praat)==pytest.approx(180,abs=.5)

def test_stream_stereo_roles_and_absolute_times():
    fs=8000;t=np.arange(fs*43)/fs
    data=np.column_stack([.5*np.sin(2*np.pi*120*t),.2*np.sin(2*np.pi*200*t)])
    c=AcousticConfig(use_reaper=False,selected_parameter_keys=('pF0','Intensity'),smooth_win_size=1)
    calls=[]
    def read(a,b):calls.append(b-a);return data[a:b]
    chunks=list(iter_parameters(read,len(data),fs,2,c,{'audio_channel':1}))
    times=np.concatenate([x[1]['Time_s'] for x in chunks]);pitch=np.concatenate([x[1]['pF0'] for x in chunks])
    assert np.all(np.diff(times)>0)
    np.testing.assert_allclose(np.diff(times),.005,atol=1e-12)
    assert np.nanmedian(pitch)==pytest.approx(200,abs=1)
    assert max(calls)<fs*24

def test_egg_cycles_and_aligned_export_share_period_values():
    fs=8000;t=np.arange(fs*2)/fs
    data=np.column_stack([.5*np.sin(2*np.pi*120*t),.2*np.sin(2*np.pi*120*t)])
    egg=dict(egg_channel=0,highpass_cutoff=25.,lowpass_cutoff=2000.,gci_method='slope',goi_method='scale',
        auto_prominence=True,peak_prominence=.01,valley_prominence=.01,silence_threshold=.01,
        storage='aligned',smooth_ms=0.,max_gap_ms=50.,derived=False)
    c=AcousticConfig(use_reaper=False,selected_parameter_keys=('pF0',),smooth_win_size=1)
    parts=list(iter_parameters(lambda a,b:data[a:b],len(t),fs,2,c,dict(audio_channel=1,egg=egg)))
    assert parts[0][0]=='egg_cycles'
    cycles=parts[0][1]
    assert len(cycles['CQ'])>150
    assert np.nanmedian(cycles['gF0'])==pytest.approx(120,abs=2)
    assert set(('CQ','SQ','gF0'))<=parts[1][1].keys()


def test_single_block_matches_existing_selected_values_and_mono_skips_egg():
    fs=8000;t=np.arange(fs*2)/fs;y=.2*np.sin(2*np.pi*160*t)
    config=AcousticConfig(use_reaper=False,selected_parameter_keys=('pF0','Intensity'),smooth_win_size=1)
    whole=analyze_audio(AudioInput(y,fs),config,selective=True).to_dataframe()
    streamed=list(iter_parameters(lambda a,b:y[a:b,None],len(y),fs,1,config,{'egg':{'egg_channel':0}}))
    assert len(streamed)==1 and streamed[0][2]['egg_skipped']=='mono_source'
    for key in whole.columns:np.testing.assert_allclose(whole[key],streamed[0][1][key],atol=1e-12,equal_nan=True)


def test_global_silence_reference_does_not_reset_at_chunk_boundary():
    fs=8000;t=np.arange(fs*43)/fs;y=.2*np.sin(2*np.pi*200*t)
    y[fs*22:]*=.001
    config=AcousticConfig(use_reaper=False,selected_parameter_keys=('pF0','Intensity'),smooth_win_size=1)
    parts=list(iter_parameters(lambda a,b:y[a:b,None],len(y),fs,1,config,{}))
    times=np.concatenate([p[1]['Time_s'] for p in parts]);pitch=np.concatenate([p[1]['pF0'] for p in parts])
    assert np.isfinite(pitch[(times>1)&(times<21)]).all()
    assert np.isnan(pitch[times>23]).all()
    assert len({p[2]['reference_intensity_db'] for p in parts})==1


def test_context_includes_long_lip_and_energy_windows():
    config=AcousticConfig(lip_smooth_win_size=1000,energy_window_ms=600)
    first,last,origin,start,stop=list(layout(60*8000,8000,config))[1]
    assert (first-origin)*.005>=5.2
