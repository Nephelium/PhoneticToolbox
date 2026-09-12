"""M09 original source comparison with fixed global-v2/local-v3 RNG sequence."""
import importlib.util
import io
from pathlib import Path
import numpy as np
import cv2
import pytest
import soundfile as sf
from phonetic_core.spec2wav import reconstruct
from ptb_worker.spec2wav_child import prepare
from ptb_worker.segmentation import unpack_bundle
from ptb_api.spec2wav_models import Spec2WavConfig

ROOT=Path(__file__).resolve().parents[2]


def old(name):
    spec=importlib.util.spec_from_file_location('original_'+name,ROOT/'phonetic_toolbox/core/spec2wav'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('seed,height,width,freq,sr',[(0,33,48,4000,44100),(27,65,90,11025,22050),(6,17,25,5000,0)])
def test_v2_fixed_seed_exact_waveform_and_actual_wav(seed,height,width,freq,sr):
    gray=np.random.RandomState(3).randint(0,256,(height,width),dtype=np.uint8)
    config=dict(time_start=.2,time_end=.35,freq_end=freq,n_iter=3,target_sr=sr,seed=seed)
    image=old('image_processing');gl=old('griffin_lim');common=old('common')
    _,magnitude,_,hop,nfft=image.load_spectrogram_image(image_data=gray,time_start=.2,time_end=.35,freq_end=freq)
    np.random.seed(seed);native=gl.griffinlim_numpy(magnitude,n_iter=3,hop_length=hop,win_length=max(1,min(int(.01*int(2*freq)),nfft)),n_fft=nfft)
    expected=common.resample_audio(native,int(2*freq),sr) if sr else native
    before=np.random.get_state();result=reconstruct(gray,**config);after=np.random.get_state()
    np.testing.assert_array_equal(result['audio'],expected);np.testing.assert_array_equal(before[1],after[1]);assert before[2:]==after[2:]
    encoded=cv2.imencode('.png',gray)[1].tobytes();bundle=unpack_bundle(prepare(encoded,config),64_000_000)
    raw=bundle.payloads[2];decoded,rate=sf.read(io.BytesIO(raw))
    original=io.BytesIO();sf.write(original,expected,sr or int(2*freq),format='WAV',subtype='PCM_16')
    assert raw==original.getvalue();assert len(decoded)==len(expected)


def test_nonzero_frequency_is_explicit_and_changes_result():
    image=np.full((32,40),255,dtype=np.uint8);image[12:14]=0
    zero=reconstruct(image,time_end=.1,n_iter=2,freq_end=4000,target_sr=0)
    band=reconstruct(image,time_end=.1,n_iter=2,freq_start=2000,freq_end=4000,target_sr=0)
    assert band['metadata']['frequency_mapping']=='band-interpolation-zero-fill-v1'
    assert band['metadata']['n_fft']==124
    assert not np.array_equal(zero['audio'],band['audio'])


def test_nonzero_band_physical_peak_matches_calibrated_frequency():
    gray=np.full((65,160),255,dtype=np.uint8);gray[31:34]=0
    result=reconstruct(gray,time_end=.5,n_iter=16,freq_start=2000,freq_end=4000,min_db=-80,target_sr=0)
    audio=result['audio'];frequencies=np.fft.rfftfreq(len(audio),1/result['sr'])
    peak=frequencies[np.argmax(abs(np.fft.rfft(audio*np.hanning(len(audio)))))]
    assert abs(peak-3000)<100,peak


def test_four_corner_warp_against_opencv_and_invalid_crossed_quad():
    gray=np.arange(80*120,dtype=np.uint8).reshape(80,120);raw=cv2.imencode('.png',gray)[1].tobytes()
    config=dict(n_iter=1,time_end=.1,corners=[dict(x=0,y=0),dict(x=1,y=0),dict(x=1,y=1),dict(x=0,y=1)])
    bundle=unpack_bundle(prepare(raw,config),64_000_000)
    restored=cv2.imdecode(np.frombuffer(bundle.payloads[0],np.uint8),0)
    points=np.float32([[0,0],[119,0],[119,79],[0,79]]);dest=np.float32([[0,0],[118,0],[118,78],[0,78]])
    np.testing.assert_array_equal(restored,cv2.warpPerspective(gray,cv2.getPerspectiveTransform(points,dest),(119,79)))
    config['corners'][1],config['corners'][2]=config['corners'][2],config['corners'][1]
    with pytest.raises(ValueError,match='invalid_image_corners'):prepare(raw,config)


@pytest.mark.parametrize('config',[dict(time_end=0),dict(time_end=31),dict(freq_start=5000,freq_end=4000),dict(seed=-1),dict(n_iter=129),dict(min_db=0,max_db=0),dict(freq_end=float('nan'))])
def test_config_bounds(config):
    with pytest.raises(ValueError):Spec2WavConfig(**config)


def test_corrupt_image_and_compute_budget():
    with pytest.raises(ValueError):prepare(b'not an image',{})
    with pytest.raises(ValueError,match='spectrogram_budget'):reconstruct(np.zeros((1000,1000),np.uint8),n_iter=128)
