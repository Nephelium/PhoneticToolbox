"""P11 practical/1: unit-based budgets and sample-derived event propagation."""
from itertools import product
import numpy as np


def same_structure(actual, reference):
    a,b=np.asarray(actual),np.asarray(reference)
    assert a.shape==b.shape, 'shape'
    assert a.dtype==b.dtype, 'dtype'
    for fn in (np.isnan,np.isposinf,np.isneginf):
        assert np.array_equal(fn(a),fn(b)), 'validity mask'
    return a,b


def continuous(actual, reference, *, atol, rtol=0):
    a,b=same_structure(actual,reference)
    finite=np.isfinite(a)
    assert np.all(np.abs(a[finite]-b[finite])<=atol+rtol*np.abs(b[finite])), 'continuous budget'


def events(actual, reference, fs):
    a,b=same_structure(np.asarray(actual,dtype=float),np.asarray(reference,dtype=float))
    assert a.ndim==1 and np.isfinite(a).all(), 'event validity'
    assert np.all(np.diff(a)>0) and np.all(np.diff(b)>0), 'event ordering'
    # Representation allowance is machine precision, not another sample.
    assert np.all(np.abs(a-b)<=1/fs+8*np.finfo(float).eps), 'event sample budget'


def derived(actual, reference, actual_events, reference_events, fs):
    """CQ=(GOI-GCI)/period; SQ=(GOI-2*peak+GCI)/(GOI-GCI).

    Require unchanged association indices and masks; evaluate the extrema of the
    reference event boxes at +/- one sample. These linear-fractional expressions
    attain their bounds at vertices when the denominator stays positive.
    """
    for a,b in zip(actual_events,reference_events):events(a,b,fs)
    for a,b in zip(actual,reference):same_structure(a,b)
    events(actual[0],reference[0],fs)
    ag,ao,ap=map(np.asarray,actual_events);bg,bo,bp=map(np.asarray,reference_events)
    for k,(g0,g1) in enumerate(zip(bg[:-1],bg[1:])):
        for arr,oldarr in ((ao,bo),(ap,bp)):
            ai=np.flatnonzero((arr>ag[k])&(arr<ag[k+1]))
            bi=np.flatnonzero((oldarr>g0)&(oldarr<g1))
            assert np.array_equal(ai,bi), 'cycle association'
        oi=np.flatnonzero((bo>g0)&(bo<g1))
        for n in (1,2):
            if not np.isfinite(reference[n][k]):continue
            assert len(oi), 'missing GOI'
            o=bo[oi[0]];sample=1/fs
            if n==1:
                assert g1-g0>2*sample, 'unstable period'
                corners=[(y-x)/(z-x) for x,y,z in product((g0-sample,g0+sample),(o-sample,o+sample),(g1-sample,g1+sample))]
                exact=(ao[oi[0]]-ag[k])/(ag[k+1]-ag[k])
            else:
                pi=np.flatnonzero((bp>g0)&(bp<o))
                api=np.flatnonzero((ap>ag[k])&(ap<ao[oi[0]]))
                assert len(pi)==1 and np.array_equal(pi,api), 'contact peak association'
                p=bp[pi[0]]
                assert o-g0>2*sample, 'unstable contact'
                corners=[(y-2*z+x)/(y-x) for x,y,z in product((g0-sample,g0+sample),(o-sample,o+sample),(p-sample,p+sample))]
                exact=(ao[oi[0]]-2*ap[pi[0]]+ag[k])/(ao[oi[0]]-ag[k])
            value=actual[n][k]
            assert abs(value-exact)<=1e-12, 'derived value inconsistent with events'
            assert min(corners)-1e-12<=value<=max(corners)+1e-12, 'derived sample envelope'


def roi_derived(actual, reference, captured_events, fs):
    """Frozen ROI metrics have no stored 100ms-path event arrays.

    Use a reverse one-sample envelope around the captured current events, require
    identical metric masks and cycle counts. The separate 50ms event path is
    checked against its own frozen events, never substituted for this path.
    """
    for a,b in zip(actual,reference):same_structure(a,b)
    events(actual[0],reference[0],fs)
    g,o,p=map(np.asarray,captured_events);s=1/fs
    assert len(actual[0])==max(0,len(g)-1)
    for k,(g0,g1) in enumerate(zip(g[:-1],g[1:])):
        oi=np.flatnonzero((o>g0)&(o<g1))
        for n in (1,2):
            if not np.isfinite(reference[n][k]):continue
            assert len(oi) and g1-g0>2*s
            goi=o[oi[0]]
            if n==1:
                corners=[(b-a)/(c-a) for a,b,c in product((g0-s,g0+s),(goi-s,goi+s),(g1-s,g1+s))]
                exact=(goi-g0)/(g1-g0)
            else:
                pi=np.flatnonzero((p>g0)&(p<goi));assert len(pi)==1 and goi-g0>2*s
                peak=p[pi[0]]
                corners=[(b-2*c+a)/(b-a) for a,b,c in product((g0-s,g0+s),(goi-s,goi+s),(peak-s,peak+s))]
                exact=(goi-2*peak+g0)/(goi-g0)
            assert abs(actual[n][k]-exact)<=1e-12
            assert min(corners)-1e-12<=reference[n][k]<=max(corners)+1e-12, 'ROI derived sample envelope'
