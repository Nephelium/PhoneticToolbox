"""R3: F0 routing, actual timestamps and unvoiced gaps before editable fill."""
from copy import deepcopy
import numpy as np
import pytest
from phonetic_core.models.audio import AudioInput
from phonetic_core.ports.acoustic import ReaperTrack
from phonetic_core.acoustic.f0_praat import compute_praat_f0_track
from phonetic_core.acoustic.alignment import align_track_to_grid
from phonetic_core.synthesis.klatt.api import defaults,extract,validate,export_parameters,import_parameters


def audio():
    t=np.arange(16000)/16000
    y=.1*np.sin(2*np.pi*150*t)
    y[(t>.35)&(t<.65)]=0
    return AudioInput(y.astype(np.float32),16000)


@pytest.mark.parametrize('method',['praat_cc','praat_ac'])
def test_actual_praat_matches_shared_time_grid_and_retains_gap(method):
    c=defaults();c['f0_method']=method;source=audio();before=source.samples.copy();info={}
    result=extract(c,source,diagnostics=info)
    track=compute_praat_f0_track(source,10,50,500,method.removeprefix('praat_'))
    expected=align_track_to_grid(track.times,track.values,np.arange(100)*10./1000.)
    measured=np.array([np.nan if v is None else v for v in info['measured_f0_hz']])
    np.testing.assert_array_equal(measured,expected)
    assert np.nanmedian(measured)==pytest.approx(150,abs=1)
    assert np.isnan(measured[40:60]).all()
    av=np.array(result['curves']['AV']['points'])
    assert not av[(av[:,0]>.4)&(av[:,0]<.6),1].any()
    assert av[1,0]==.01 and av[-1,0]==1
    assert info['actual_f0_backend']==method and info['extraction_revision']=='m06-extract/2'
    np.testing.assert_array_equal(source.samples,before)


def test_reaper_port_is_explicit_and_keeps_missing_frames():
    c=defaults();c['f0_method']='reaper';info={};calls=[]
    def backend(source,step,lo,hi,**kw):
        calls.append((step,lo,hi,kw))
        return ReaperTrack(np.array([.1,.2,.3,.4]),np.array([120.,130.,np.nan,140.]),'test_port')
    result=extract(c,audio(),reaper=backend,diagnostics=info)
    assert calls==[(.01,50.,500.,dict(hilbert=False,no_highpass=False))]
    assert info['actual_f0_backend']=='test_port'
    assert info['measured_f0_hz'][20]==130
    assert info['measured_f0_hz'][30] is None
    assert not info['voiced_mask'][30]
    assert result['curves']['AV']['points'][30][1]==0
    with pytest.raises(ValueError,match='m06_reaper_unavailable'):extract(c,audio())
    def bad(*a,**k):raise RuntimeError('test')
    with pytest.raises(ValueError,match='m06_reaper_failed'):extract(c,audio(),reaper=bad)


def test_old_config_defaults_nonmutating_and_invalid_method_rejected():
    old=defaults();del old['f0_method'];before=deepcopy(old)
    assert validate(old)['f0_method']=='praat_cc' and old==before
    for method in ('praat_cc','praat_ac','reaper'):
        c=defaults();c['f0_method']=method
        assert import_parameters(export_parameters(c))==c
    for method in (None,'unknown',{},False):
        with pytest.raises(ValueError,match='m06_invalid_f0_method'):validate(dict(old,f0_method=method))


def test_shimmer_measurement_percent_becomes_stored_fraction(monkeypatch):
    import phonetic_core.synthesis.klatt.engine as module
    # Independent unit contract: APQ5=.5 percent must be .005 in the curve.
    monkeypatch.setattr(module,'compute_jitter_shimmer',lambda *a,**k:dict(
        Jitter_PPQ5=np.full(100,.5),Shimmer_APQ5=np.full(100,.5)))
    result=extract(defaults(),audio())
    assert all(v==.005 for _,v in result['curves']['Shimmer']['points'])
    assert all(v==.5 for _,v in result['curves']['Jitter']['points'])
