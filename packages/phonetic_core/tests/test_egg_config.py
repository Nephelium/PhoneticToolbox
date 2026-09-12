"""M03 public array boundary: explicit inputs, cancellation, metadata and errors."""
import dataclasses
import threading

import numpy as np
import pytest

from phonetic_core.egg import EGGConfig, prepare, analyze_events, cq_segment, events_segment
from phonetic_core.egg.errors import EggError, EggCancelled
from phonetic_core.egg.f0 import praat_pitch
from phonetic_core.egg.inverse import inverse_filter


@pytest.mark.parametrize('kwargs', [
    {'gci_method':'unknown'}, {'goi_method':'unknown'}, {'criterion_level':.75},
    {'peak_prominence':-1}, {'valley_prominence':float('nan')},
    {'highpass_cutoff':0}, {'lowpass_cutoff':-2}, {'spec_window_ms':0},
    {'spec_vmin':-10,'spec_vmax':-70}, {'auto_prominence':'yes'},
    {'min_auto_prominence':float('inf')}, {'if_order_heuristic_add':7},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(EggError):
        EGGConfig(**kwargs)


def test_config_is_immutable_and_ui_defaults_are_explicit():
    assert EGGConfig().goi_method == 'slope'
    assert EGGConfig.for_workbench().goi_method == 'scale'
    with pytest.raises(dataclasses.FrozenInstanceError):
        EGGConfig().goi_method = 'scale'


@pytest.mark.parametrize('samples', [np.zeros(100), np.zeros((100,3)), np.zeros((0,2)),
                                    np.full((100,2),np.nan), np.full((100,2),np.inf),
                                    np.array([['x','y']]), np.zeros((10,2), dtype=bool)])
def test_invalid_samples_rejected(samples):
    with pytest.raises(EggError):
        prepare(samples, 44100, EGGConfig())


@pytest.mark.parametrize('fs', [0,-1,True,44100.5,float('nan')])
def test_invalid_sample_rate(fs):
    with pytest.raises(EggError):
        prepare(np.zeros((100,2)), fs, EGGConfig())


def test_input_ownership_normalization_duration_and_no_global_mutation():
    samples = np.column_stack((np.arange(2000,dtype=np.int16), -np.arange(2000,dtype=np.int16)))
    old = samples.copy()
    cfg = EGGConfig.for_workbench()
    result = prepare(samples, 16000, cfg)
    np.testing.assert_array_equal(samples, old)
    assert not np.shares_memory(samples, result.egg_signal_raw)
    assert np.max(np.abs(result.egg_signal_raw)) == np.float32(.7)
    assert result.sample_duration_s == 2000/16000
    assert result.last_sample_time_s == 1999/16000
    assert result.file_duration == result.last_sample_time_s  # explicitly legacy property
    updated = analyze_events(result, cfg)
    assert updated is not result
    assert result.gci_times == []


def test_short_filter_failure_is_explicit():
    with pytest.raises(EggError, match='filter_failed'):
        prepare(np.zeros((8,2)), 44100, EGGConfig())


def test_filter_nyquist_is_rejected_without_clamping():
    with pytest.raises(EggError, match='filter_cutoff'):
        prepare(np.zeros((100,2)), 1000, EGGConfig())


def test_pre_cancel_and_in_calculation_cancel_are_not_empty_success():
    event = threading.Event(); event.set()
    with pytest.raises(EggCancelled):
        prepare(np.zeros((100,2)), 44100, EGGConfig(), cancel_event=event)
    with pytest.raises(EggCancelled):
        praat_pitch(np.zeros(4410),44100,cancel_event=event)
    with pytest.raises(EggCancelled):
        inverse_filter(np.zeros(4410),44100,np.array([]),cancel_event=event)


@pytest.mark.parametrize('bounds', [(float('nan'),1),(0,float('inf')),(2,1)])
def test_invalid_roi_rejected(bounds):
    state=prepare(np.zeros((1000,2)),44100,EGGConfig())
    for fn in (cq_segment,events_segment):
        with pytest.raises(EggError):
            fn(state,*bounds,EGGConfig())


@pytest.mark.parametrize('order', [0,-1,1.5,True,10000])
def test_invalid_or_unavailable_inverse_filter_is_explicit(order):
    with pytest.raises(EggError):
        inverse_filter(np.zeros(1000),44100,np.array([.001,.01,.02]),lp_order=order)


def test_missing_gci_and_short_praat_are_explicit():
    with pytest.raises(EggError, match='inverse_unavailable'):
        inverse_filter(np.zeros(1000),44100,np.array([]))
    with pytest.raises(EggError, match='pitch_failed'):
        praat_pitch(np.zeros(8),44100)


@pytest.mark.parametrize('operation', ['global','cq','events','inverse'])
def test_cancellation_polled_inside_actual_numerical_loops(operation):
    class CancelAfterPolls:
        def __init__(self): self.calls=0
        def is_set(self):
            self.calls+=1
            return self.calls>=5
    t=np.arange(16000)/16000
    samples=np.column_stack((np.sin(2*np.pi*120*t),np.sin(2*np.pi*120*t)))
    cfg=EGGConfig.for_workbench()
    state=prepare(samples,16000,cfg)
    cancel=CancelAfterPolls()
    with pytest.raises(EggCancelled):
        if operation=='global': analyze_events(state,cfg,cancel)
        elif operation=='cq': cq_segment(state,.1,.8,cfg,cancel_event=cancel)
        elif operation=='events': events_segment(state,.1,.8,cfg,cancel_event=cancel)
        else: inverse_filter(state.audio_signal,16000,np.arange(.01,.9,.01),lp_order=12,cancel_event=cancel)
    assert cancel.calls>=5
    assert state.gci_times==[]


def test_config_snapshot_is_preserved_with_signal_and_event_results():
    cfg=EGGConfig.for_workbench()
    state=prepare(np.zeros((1000,2)),44100,cfg)
    updated=analyze_events(state,dataclasses.replace(cfg,peak_prominence=.03))
    assert state.preprocessing_config==cfg
    assert state.analysis_config is None
    assert updated.analysis_config.peak_prominence==.03
    assert updated.preprocessing_config==cfg
