"""Scientific boundaries for web M16, with synthetic in-memory WAV inputs."""
import io
import numpy as np
import pytest
import soundfile as sf
from ptb_worker.recording_child import compute,load
from ptb_worker.recording_preview import display


def wav(x,rate=48000):
    out=io.BytesIO();sf.write(out,x,rate,format='WAV',subtype='FLOAT');return out.getvalue()


def config(action,**kwargs):
    return dict(action=action,roles=['microphone','egg'],channels=[0],channel=0,gain_db=6.,strength=1.,subtype='FLOAT')|kwargs


def test_gain_preserves_egg_and_original_float_scale():
    x=np.array([[.25,-.9],[-.6,.2],[1.1,.7]],dtype=np.float32);before=x.copy()
    raw,meta=compute([wav(x)],config('gain'));y,rate=load(raw)
    np.testing.assert_array_equal(x,before);np.testing.assert_array_equal(y[:,1],x[:,1])
    np.testing.assert_allclose(y[:,0],x[:,0]*10**(6/20),rtol=2e-7,atol=0)
    assert rate==48000 and meta['algorithm']=='linear-gain/1' and y[2,0]>1


def test_all_channel_half_open_edit_and_paste_without_resampling():
    x=np.column_stack((np.arange(10),-np.arange(10))).astype(np.float32)/20
    source=wav(x)
    kept,_=compute([source],config('keep',start=2,end=5));np.testing.assert_array_equal(load(kept)[0],x[2:5])
    cut,_=compute([source],config('cut',start=2,end=5));np.testing.assert_array_equal(load(cut)[0],np.concatenate([x[:2],x[5:]]))
    pasted,_=compute([source],config('paste',start=1,end=1,reference_start=6,reference_end=8,reference_roles=['microphone','egg']))
    np.testing.assert_array_equal(load(pasted)[0],np.concatenate([x[:1],x[6:8],x[1:]]))
    with pytest.raises(ValueError,match='同采样率'):compute([source,wav(x,44100)],config('paste',start=1,reference_start=0,reference_end=3))
    with pytest.raises(ValueError,match='空录音'):compute([source],config('delete',start=0,end=10))


def test_noise_and_denoise_retain_unselected_channel_and_frame_count():
    rng=np.random.default_rng(612);n=300000
    noise=rng.normal(0,.03,n).astype(np.float32)
    x=np.column_stack([noise.copy(),np.sin(np.arange(n)*.02)*.2]).astype(np.float32)
    x[10000:,0]+=.2*np.sin(np.arange(n-10000)*2*np.pi*210/48000)
    source=wav(x)
    _,meta=compute([source],config('noise',start=0,end=5000));assert meta['algorithm']=='ptb-spectral-subtraction/1'
    output,_=compute([source],config('denoise',reference_start=0,reference_end=5000))
    y,_=load(output);assert y.shape==x.shape
    np.testing.assert_array_equal(y[:,1],x[:,1]);assert np.isfinite(y).all()
    assert np.mean(y[:5000,0]**2)<np.mean(x[:5000,0]**2)
    with pytest.raises(ValueError,match='麦克风'):compute([source],config('noise',start=0,end=5000,**{'channel':1}))


def test_praat_display_uses_exact_same_pixels_and_view_grid():
    from phonetic_core.recording.praat_display import PraatDisplaySpectrum
    t=np.arange(12000)/48000;x=np.column_stack([.2*np.sin(2*np.pi*220*t),.3*np.sin(2*np.pi*880*t)]).astype(np.float32)
    direct=PraatDisplaySpectrum(10000,48000,1,320);direct.add(x[1000:11000]);expected=direct.result()
    actual=display(wav(x),start=1000,end=11000,channel=1,width=320)
    import base64
    assert base64.b64decode(actual['pixels_base64'])==bytes(p for row in expected['pixels'] for p in row)
    assert actual['times']==expected['times'] and actual['time_edges']==expected['time_edges']
    assert actual['backend']=='praat' and actual['display_revision']=='m16-praat-display/1'


@pytest.mark.parametrize('subtype',['PCM_16','PCM_24','FLOAT'])
def test_export_explicit_subtype_range_and_roundtrip(subtype):
    x=np.column_stack([np.linspace(-.5,.5,400),np.linspace(.2,-.2,400)]).astype(np.float32)
    out,meta=compute([wav(x)],dict(config('export',start=30,end=250),subtype=subtype))
    info=sf.info(io.BytesIO(out));assert (info.frames,info.channels,info.samplerate,info.subtype)==(220,2,48000,subtype)
    np.testing.assert_allclose(load(out)[0],x[30:250],atol=2**(-15 if subtype=='PCM_16' else -23),rtol=0)
    if subtype!='FLOAT':
        with pytest.raises(ValueError,match='量化范围'):compute([wav(x*5)],dict(config('export'),subtype=subtype))
