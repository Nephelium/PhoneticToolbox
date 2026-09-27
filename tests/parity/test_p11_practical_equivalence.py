"""Separate Linux practical gate; original Windows exact tests remain intact."""
import json
from pathlib import Path
import numpy as np
import pytest
from practical_equivalence import continuous, events, derived, roi_derived
from phonetic_core.egg import EGGConfig, prepare, analyze_events, cq_segment, events_segment
from phonetic_core.egg.metrics import calculate_cq_sq
from phonetic_core.egg.f0 import praat_pitch
from phonetic_core.egg.inverse import inverse_filter
from phonetic_core.lpc import LPCConfig, compute_spectrum

ROOT=Path(__file__).resolve().parents[1]/'fixtures'


@pytest.mark.parametrize('index',range(8))
def test_egg_events_and_derived_values(index):
    meta=json.loads((ROOT/'m03/EGG-SYN-PCM16.json').read_text('utf-8'))
    with np.load(ROOT/'m03/EGG-SYN-PCM16.npz') as data:
        config=EGGConfig(**meta['variants'][index]['config'])
        samples=np.column_stack([data['load.egg_signal_raw'],data['load.audio_signal']])
        result=analyze_events(prepare(samples,44100,config),config)
        continuous(result.egg_signal_processed,data['load.egg_signal_processed'],atol=1e-6)
        old=meta['variants'][index]['events']
        new=[result.gci_times,result.goi_times,result.peak_times]
        derived(calculate_cq_sq(*new),calculate_cq_sq(*old),new,old,44100)
        frozen=meta['variants'][index]['gci_f0']
        expected=[data[v['array']] if isinstance(v,dict) else v for v in frozen]
        if expected[0] is None:
            assert result.gci_f0_times is None and result.gci_f0_values is None
        else:
            from practical_equivalence import same_structure
            events(result.gci_f0_times,expected[0],44100)
            a,b=same_structure(result.gci_f0_values,expected[1]);mask=np.isfinite(b)
            period=1/b[mask];delta=2/44100
            assert np.all(period>delta)
            assert np.all(a[mask]>=1/(period+delta)-1e-9)
            assert np.all(a[mask]<=1/(period-delta)+1e-9)


def test_praat_units_and_inverse_waveforms():
    meta=json.loads((ROOT/'m03/EGG-SYN-PCM16.json').read_text('utf-8'))
    with np.load(ROOT/'m03/EGG-SYN-PCM16.npz') as data:
        track=praat_pitch(data['load.audio_signal'],44100)
        continuous(track.times,data['praat.actual.times'],atol=0)
        continuous(track.values,data['praat.legacy.values'],atol=1e-6)
        variant=next(v for v in meta['variants'] if v['config']['gci_method']=='slope' and v['config']['goi_method']=='scale' and v['config']['auto_prominence'])
        audio=data['load.audio_signal'][:5292]
        gci=np.asarray([t for t in variant['events'][0] if t<len(audio)/44100])
        for order,key in ((None,'auto'),(12,'explicit')):
            continuous(inverse_filter(audio,44100,gci,lp_order=order),data['inverse.'+key],atol=1e-8,rtol=1e-6)


@pytest.mark.parametrize('index',range(8))
def test_both_roi_policies_against_their_own_frozen_values(index,monkeypatch):
    import phonetic_core.egg._legacy as legacy
    captured=[]
    original=legacy.calculate_cq_sq
    def observe(*args):
        captured.append(args)
        return original(*args)
    monkeypatch.setattr(legacy,'calculate_cq_sq',observe)
    meta=json.loads((ROOT/'m03/EGG-SYN-PCM16.json').read_text('utf-8'))
    with np.load(ROOT/'m03/EGG-SYN-PCM16.npz') as data:
        config=EGGConfig(**meta['variants'][index]['config'])
        state=prepare(np.column_stack([data['load.egg_signal_raw'],data['load.audio_signal']]),44100,config)
        for modes in meta['variants'][index]['roi'].values():
            for mode,item in modes.items():
                for a,b in zip(events_segment(state,*item['bounds_s'],config,use_raw_signal=mode=='raw'),item['events']['value']):
                    if a is None or b is None:assert a is b
                    else:events(a,b,44100)
                captured.clear()
                actual=cq_segment(state,*item['bounds_s'],config,use_raw_signal=mode=='raw')
                old=[data[v['array']] if isinstance(v,dict) else v for v in item['cq']['value']]
                if old[0] is None:assert actual==(None,None,None)
                else:
                    assert len(captured)==1
                    roi_derived(actual,old,captured[0],44100)


LPC=json.loads((ROOT/'m04/result.json').read_text('utf-8'))
@pytest.mark.parametrize('case',list(LPC['cases']))
def test_lpc_units_and_error_behavior(case):
    config=LPC['cases'][case]
    with np.load(ROOT/'m04/arrays.npz') as data:
        audio=data[case+'.input'].copy();before=audio.tobytes()
        if config['status']=='error':
            with pytest.raises(ValueError):compute_spectrum(audio,config['rate'],LPCConfig(**config['config']))
        else:
            result=compute_spectrum(audio,config['rate'],LPCConfig(**config['config']))
            continuous(result.frequencies_hz,data[case+'.frequency'],atol=0)
            continuous(result.magnitude_db,data[case+'.db'],atol=1e-6)
            continuous(np.array([result.amp_min_db,result.amp_max_db],dtype=float),np.array(config['y_range'],dtype=float),atol=1e-6)
        assert audio.tobytes()==before


@pytest.mark.parametrize('bad',[[0,.011,.02],[0,.01],[0,.01,.01],[0,float('nan'),.02]])
def test_event_checker_rejects_out_of_budget_missing_duplicate_and_invalid(bad):
    with pytest.raises(AssertionError):events(bad,[0,.01,.02],44100)


def test_checker_rejects_mask_and_excess_continuous_error():
    for values in (np.array([np.nan,1.]),np.array([0.,1.001])):
        with pytest.raises(AssertionError):continuous(values,np.array([0.,1.]),atol=1e-6)


def test_one_sample_does_not_license_arbitrary_cq_sq():
    old=[[0,.01,.02],[.006,.016],[.002,.012]]
    new=[[0,.01+1/44100,.02],[.006,.016],[.002,.012]]
    actual=calculate_cq_sq(*new);expected=calculate_cq_sq(*old)
    derived(actual,expected,new,old,44100)
    actual[2][0]+=.001
    with pytest.raises(AssertionError):derived(actual,expected,new,old,44100)
