"""M09-R1 independent numeric and geometry acceptance fixtures."""
import numpy as np
import pytest
from phonetic_core.spec2wav.editing import edit_audio, analyze_audio, paint, rectify


def stroke(color=0,opacity=1,points=None,size=5):
    return dict(color=color,opacity=opacity,size=size,points=points or [dict(x=.5,y=.5)])


@pytest.mark.parametrize('sr,n,fft',[(8000,2,512),(16000,16001,1024),(44100,47001,2048),(96000,96017,1024)])
def test_unpainted_roundtrip_keeps_last_sample_channels_and_timing(sr,n,fft):
    audio=np.random.default_rng(91).normal(0,.1,(n,2));audio[-1]=[.61,-.39]
    result=edit_audio(audio,sr,[],n_fft=fft)
    np.testing.assert_allclose(result['audio'][:,0],audio[:,0],atol=4e-15,rtol=0)
    np.testing.assert_array_equal(result['audio'][:,1],audio[:,1])
    assert result['sr']==sr and result['metadata']['samples']==n
    assert result['metadata']['n_iter']==0 and result['metadata']['changed_bins']==0


def test_black_white_opacity_and_once_per_stroke():
    gray=np.full((40,60),100,dtype=np.uint8)
    a,touched=paint(gray,[stroke(0,.25)])
    b,_=paint(gray,[stroke(255,.5)])
    assert a[20,30]==75 and b[20,30]==177.5
    assert np.all(a[~touched]==100)
    repeated,_=paint(gray,[stroke(0,.25,points=[dict(x=.5,y=.5)]*100)])
    np.testing.assert_array_equal(a,repeated)
    invisible,touched=paint(gray,[stroke(0,0)])
    np.testing.assert_array_equal(invisible,gray);assert not touched.any()


def test_black_draws_local_frequency_and_keeps_other_channel():
    sr=16000;n=16001;t=np.arange(n)/sr
    audio=np.column_stack([.2*np.sin(2*np.pi*400*t),.1*np.sin(2*np.pi*1700*t)])
    before=audio.copy();s=stroke(0,1,[dict(x=.35,y=.5),dict(x=.65,y=.5)],7)
    result=edit_audio(audio,sr,[s]);np.testing.assert_array_equal(audio,before)
    np.testing.assert_array_equal(result['audio'][:,1],audio[:,1])
    f=np.fft.rfftfreq(n,1/sr);band=(f>3900)&(f<4100)
    assert np.linalg.norm(np.fft.rfft(result['audio'][:,0])[band])>100*np.linalg.norm(np.fft.rfft(audio[:,0])[band])
    assert result['metadata']['phase_method']=='original-stft-phase'


def test_silence_is_finite_and_black_can_add_energy():
    result=edit_audio(np.zeros(16000),16000,[stroke(size=20)])
    assert np.isfinite(result['audio']).all() and np.max(abs(result['audio']))>0
    assert result['metadata']['zero_magnitude_phase']=='zero-radians'


def test_skewed_quad_maps_named_corner_markers_to_rectangle():
    import cv2
    gray=np.full((101,121),255,np.uint8)
    points=[(10,5),(110,25),(90,90),(20,80)]
    for i,p in enumerate(points):cv2.circle(gray,p,4,30+i*50,-1)
    corners=[dict(x=x/120,y=y/100) for x,y in points]
    corrected=rectify(gray,corners)
    assert corrected.shape!=gray.shape
    assert [int(corrected[0,0]),int(corrected[0,-1]),int(corrected[-1,-1]),int(corrected[-1,0])]==[30,80,130,180]
    assert np.all(gray[0]==255)
    with pytest.raises(ValueError,match='invalid_image_corners'):rectify(gray,[corners[0],corners[2],corners[1],corners[3]])


@pytest.mark.parametrize('audio,sr,kw',[(np.zeros(10),4000,{}),(np.zeros((20,3)),16000,{}),(np.zeros(480001),16000,{}),(np.full(10,np.nan),16000,{}),(np.zeros(10),16000,{'channel':1})])
def test_audio_bounds(audio,sr,kw):
    with pytest.raises(ValueError,match='invalid_spectrogram_audio'):analyze_audio(audio,sr,**kw)
