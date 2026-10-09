"""M07-R1 display expectations use analytic constant-F0 stimuli."""
import copy
import numpy as np
import pytest
from phonetic_core.manipulation.m07_display import synthesis_f0_display


def snapshot(kind=3, reverse=False, alignment='normalize'):
    return dict(action='generate', algorithm='m07-v2-lpc-residual/1',
                generation=dict(step_count=3), source_f0=[120.] * 501,
                target_f0=[180.] * 701, source_bounds=[0, 5512],
                target_bounds=[0, 7717], source_samples=5513, target_samples=7718,
                sample_rate_hz=11025, continuum_type=kind,
                reverse_direction=reverse, alignment=alignment)


@pytest.mark.parametrize('kind', [1, 2, 3])
@pytest.mark.parametrize('reverse', [False, True])
def test_step_curves_and_direction_with_unequal_duration(kind, reverse):
    meta=snapshot(kind, reverse);before=copy.deepcopy(meta)
    display=synthesis_f0_display(meta)
    expected=([180.,180.,180.] if reverse else [120.,120.,120.]) if kind==1 else ([180.,150.,120.] if reverse else [120.,150.,180.])
    assert [curve['name'] for curve in display['curves']]==['step01','step02','step03']
    for curve, hz in zip(display['curves'],expected):
        assert curve['axis'][0]==0 and curve['axis'][-1]==100
        np.testing.assert_allclose(curve['values'],hz,rtol=0,atol=1e-10)
    assert meta==before


def test_onset_uses_each_step_duration_instead_of_combined_timeline():
    for reverse,end in [(False,500),(True,700)]:
        display=synthesis_f0_display(snapshot(reverse=reverse,alignment='onset'))
        assert all(curve['axis'][-1]==end for curve in display['curves'])


def test_nonconstant_target_uses_original_fft_resampling():
    # One complete cosine period has an analytic resampling at any grid length.
    meta=snapshot();meta.update(source_f0=[120.]*101,
        target_f0=(180+20*np.cos(2*np.pi*np.arange(201)/201)).tolist(),
        source_bounds=[0,1102],target_bounds=[0,2205],source_samples=1103,target_samples=2206)
    curves=synthesis_f0_display(meta)['curves']
    expected=180+20*np.cos(2*np.pi*np.arange(101)/101)
    np.testing.assert_allclose(curves[2]['values'],expected,rtol=0,atol=1e-10)
    np.testing.assert_allclose(curves[1]['values'],(120+expected)/2,rtol=0,atol=1e-10)


def test_display_budget_and_nonfinite_rejection():
    meta=snapshot();meta['generation']['step_count']=50
    assert len(synthesis_f0_display(meta)['curves'])==50
    for key,value in [('source_f0',[float('nan')]*501),('source_f0',[120.]*10003),('source_bounds',[-1,5512]),('algorithm','unknown')]:
        meta=snapshot();meta[key]=value
        with pytest.raises(ValueError):synthesis_f0_display(meta)
