import numpy as np
from numpy.testing import assert_allclose
from phonetic_core.egg import EGGConfig
from phonetic_core.egg.bounded import scan, prepare, analyze_events
from phonetic_core.egg._inverse_legacy import autocorrelation


def test_global_reference_and_local_reads_do_not_normalize_each_block():
    fs=8000;frames=fs*43;reads=[]
    def read(a,b):
        reads.append(b-a);t=np.arange(a,b)/fs
        return np.column_stack((.002*t+.1*np.sin(2*np.pi*150*t),.1*np.cos(2*np.pi*100*t)))
    reference=scan(read,frames,fs)
    result=prepare(read,frames,fs,reference,EGGConfig(),flip_channels=False)
    expected=read(0,frames).astype(np.float32)
    assert_allclose(reference['peaks'],np.max(np.abs(expected),axis=0))
    assert_allclose(result.egg_signal_raw[fs*40:fs*41],expected[fs*40:fs*41,0]/reference['peaks'][0]*.7,rtol=2e-7)
    reads.clear();one=result.egg_signal_processed[fs*20-200:fs*20+200]
    two=result.egg_signal_processed[fs*20-200:fs*20+200]
    assert_allclose(one,two,rtol=0,atol=0)
    assert max(reads)<=fs*40
    assert len(result.egg_signal_processed.context.cache)<=3
    assert result.time_vector[-1]==(frames-1)/fs


def test_events_have_unique_global_ownership_across_twenty_second_boundary():
    fs=8000;frames=fs*42
    def read(a,b):
        t=np.arange(a,b)/fs
        return np.column_stack((np.sin(2*np.pi*150*t),np.cos(2*np.pi*150*t)))
    result=prepare(read,frames,fs,scan(read,frames,fs),EGGConfig())
    result=analyze_events(result,EGGConfig())
    events=np.array(result.gci_times)
    assert np.all(np.diff(events)>0)
    assert np.max(np.diff(events[(events>19.9)&(events<20.1)]))<.007
    assert events[-1]>41.9


def test_long_autocorrelation_retains_only_requested_mathematical_lags():
    y=np.random.default_rng(23).normal(size=60001)
    expected=np.array([sum(y[i]*y[i+k] for i in range(len(y)-k)) for k in range(4)])
    actual=autocorrelation(y,3)
    assert actual.shape==(4,)
    assert_allclose(actual,expected,rtol=2e-13,atol=1e-10)


def test_overlap_crop_matches_independent_whole_sos_filter_away_from_file_edges():
    from scipy import signal
    fs=8000;frames=fs*43;t=np.arange(frames)/fs
    values=np.column_stack((.002*t+.1*np.sin(2*np.pi*151.3*t),.1*np.cos(2*np.pi*100*t))).astype(np.float32)
    cfg=EGGConfig();reference=scan(lambda a,b:values[a:b],frames,fs)
    result=prepare(lambda a,b:values[a:b],frames,fs,reference,cfg)
    y=(values[:,0].astype(float)-reference['intercepts'][0]-reference['slopes'][0]*t)*.7/reference['peaks'][0]
    for frequency,kind in [(cfg.highpass_cutoff,'high'),(cfg.lowpass_cutoff,'low')]:
        y=signal.sosfiltfilt(signal.butter(4,frequency/(fs/2),btype=kind,output='sos'),y)
    assert_allclose(result.egg_signal_processed[fs*20-100:fs*20+100],y[fs*20-100:fs*20+100],rtol=0,atol=2e-10)
    assert result.egg_signal_raw[-1]==result.egg_signal_raw[frames-1]


def test_low_cutoff_halo_is_bounded_and_stable_at_96khz():
    from phonetic_core.egg.bounded import Context
    cfg=EGGConfig(highpass_cutoff=.5)
    context=Context(lambda a,b:np.zeros((b-a,2)),96000*100,96000,dict(peaks=np.ones(2),slopes=np.zeros(2),intercepts=np.zeros(2)),cfg)
    assert 96000<context.padding<=96000*30


def test_low_cutoff_local_waveforms_and_events_remain_bounded_at_96khz():
    from phonetic_core.egg import cq_segment, events_segment
    from phonetic_core.egg.preview import micro_waveforms
    fs=96000;cfg=EGGConfig(highpass_cutoff=1)
    def read(a,b):
        t=np.arange(a,b)/fs
        return np.column_stack((np.sin(2*np.pi*150*t),np.cos(2*np.pi*150*t)))
    reference=dict(peaks=np.ones(2),slopes=np.zeros(2),intercepts=np.zeros(2))
    result=prepare(read,fs*41,fs,reference,cfg)
    times,audio,egg=micro_waveforms(result,cfg,20,2000)
    assert len(times)==fs*2 and np.isfinite(egg).all() and np.max(np.abs(egg))<2
    gci,goi,peaks=events_segment(result,19,21,cfg)
    owned=np.array(gci);owned=owned[(owned>=19)&(owned<21)]
    assert 290<len(owned)<310
    assert abs(np.median(np.diff(owned))-1/150)<2/fs
    cq_times,cq,sq=cq_segment(result,19,21,cfg)
    assert len(cq_times)>290 and np.isfinite(cq).sum()>290
