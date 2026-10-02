import numpy as np
import pytest
from phonetic_core.recording import meter,noise_profile,denoise,gain_audio,select,remove,insert,frames

def test_meter_raw_and_digital_clipping_separate():
    x=np.array([[.9,-1.0],[.1,.5]],np.float32);m=meter(x,6)
    assert m['raw_clip']==[0,1];assert m['digital_clip']==[1,1];assert m['near_full_scale']==[0,1]
    assert np.array_equal(x,np.array([[.9,-1.],[.1,.5]],np.float32))

@pytest.mark.parametrize('a,b',[(-1,2),(0,11),(5,4),(1.1,2),(True,2)])
def test_invalid_selection(a,b):
    with pytest.raises(ValueError):select([{'start':0,'end':10}],a,b)

def test_selection_across_boundaries_exact():
    s=[{'file':'a','start':10,'end':20},{'file':'b','start':5,'end':15}]
    assert select(s,9,11)==[{'file':'a','start':19,'end':20},{'file':'b','start':5,'end':6}]
    assert frames(remove(s,0,1))==19;assert frames(insert(s,0,select(s,0,1)))==21;assert s[0]['start']==10

def test_noise_rejects_silence_short_and_nonfinite():
    for x in (np.zeros(4000),np.ones(100),np.full(4000,np.nan)):
        with pytest.raises(ValueError):noise_profile(x)

def test_denoise_keeps_all_unselected_channels_exact():
    rng=np.random.default_rng(5);x=rng.normal(0,.02,(12000,3)).astype(np.float32);original=x.copy();profile,_=noise_profile(x[:4000,0]);y=denoise(x,profile,[0]);assert np.array_equal(y[:,1:],x[:,1:]);assert np.array_equal(original,x);assert np.isfinite(y).all();assert y.shape==x.shape

def test_gain_is_explicit_and_does_not_touch_egg():
    x=np.ones((10,2),np.float32)*.5;y=gain_audio(x,6,[0]);assert np.array_equal(x[:,1],y[:,1]);assert y[0,0]>.99;assert x[0,0]==.5

def test_dual_audio_channels_both_processed_when_explicitly_selected():
    rng=np.random.default_rng(11);x=rng.normal(0,.01,(8000,2)).astype(np.float32);profile,_=noise_profile(x[:4000,0]);y=denoise(x,profile,[0,1]);assert not np.array_equal(x[:,0],y[:,0]);assert not np.array_equal(x[:,1],y[:,1]);z=gain_audio(x,6,[0,1]);assert np.allclose(z,x*10**.3);assert y.shape==x.shape
