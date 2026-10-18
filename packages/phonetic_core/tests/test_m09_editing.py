"""M09-R1 independent numeric and geometry acceptance fixtures."""
import numpy as np
import pytest
from phonetic_core.spec2wav.editing import edit_audio, analyze_audio, paint, rectify


@pytest.mark.parametrize('size',[1,2,3,74,200])
def test_pixel_diameter_and_legacy_footprint(size):
    gray=np.full((401,401),255,np.uint8)
    mark=dict(color=0,opacity=1,size=size,size_unit='px',points=[dict(x=.5,y=.5)])
    output,touched=paint(gray,[mark]);ys,xs=np.where(touched)
    assert xs.max()-xs.min()+1==size and ys.max()-ys.min()+1==size
    assert np.all(output[touched]==0)
    if size==1:
        assert touched.sum()==1
        legacy=dict(mark);legacy.pop('size_unit')
        assert paint(gray,[legacy])[1].sum()==5


def stroke(color=0,opacity=1,points=None,size=5):
    return dict(color=color,opacity=opacity,size=size,points=points or [dict(x=.5,y=.5)])


def test_wide_drawing_pixel_coverage_maps_to_original_time_cells():
    gray=np.full((9,4),140,np.uint8)
    mark=dict(color=0,opacity=.5,size=1,size_unit='px',canvas_width=7,points=[dict(x=.5,y=.5)])
    output,touched=paint(gray,[mark])
    # Drawing pixel [3,4) overlaps original cells [1.75,3.5), [3.5,5.25)
    # by half a pixel each: .5 / 1.75 = 2/7 coverage, times .5 opacity.
    expected=gray.astype(float);expected[4,1:3]=140*(1-.5*(.5/1.75))
    np.testing.assert_array_equal(output,expected)
    assert touched.sum()==2 and touched[4,1] and touched[4,2]


def test_wide_brush_preserves_original_stft_grid_phase_and_other_channel():
    t=np.arange(9217)/16000
    audio=np.column_stack([.2*np.sin(2*np.pi*400*t),.1*np.cos(2*np.pi*1700*t)])
    mark=dict(color=0,opacity=.5,size=1,size_unit='px',canvas_width=1024,points=[dict(x=.5,y=.3)])
    result=edit_audio(audio,16000,[mark])
    assert result['target'].shape==(513,38) and result['metadata']['changed_bins']>0
    assert result['audio'].shape==audio.shape and result['metadata']['phase_method']=='original-stft-phase'
    np.testing.assert_array_equal(result['audio'][:,1],audio[:,1])


def test_drawing_canvas_budget_is_bounded():
    mark=dict(color=0,opacity=1,size=12,size_unit='px',canvas_width=32768,points=[dict(x=.5,y=.5)])
    with pytest.raises(ValueError,match='spectrogram_budget'):paint(np.zeros((513,38),np.uint8),[mark])


def test_display_band_and_height_map_only_to_selected_frequency_cells():
    gray=np.full((8,8),200,np.uint8)
    mark=dict(color=0,opacity=1,size=1,size_unit='px',canvas_width=8,canvas_height=4,
              frequency_fraction=.5,points=[dict(x=3/7,y=1/3)])
    output,touched=paint(gray,[mark])
    expected=gray.astype(float);expected[5,3]=0
    np.testing.assert_array_equal(output,expected)
    assert touched.sum()==1 and touched[5,3]
    assert not touched[:4].any()


def test_display_resize_preserves_frequency_time_location_for_same_mark():
    gray=np.full((12,12),200,np.uint8)
    def marker(width,height,x,y):
        return dict(color=0,opacity=1,size=1,size_unit='px',canvas_width=width,canvas_height=height,
                    frequency_fraction=.5,points=[dict(x=x,y=y)])
    # Both drawing pixel centers fall in source cell x=3, y=8 despite resize.
    _,a=paint(gray,[marker(24,12,6/23,4/11)])
    _,b=paint(gray,[marker(36,18,9/35,6/17)])
    assert np.argwhere(a).tolist()==[[8,3]] and np.argwhere(b).tolist()==[[8,3]]


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
