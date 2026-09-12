"""Real installed-core exports against frozen V2 and independent file readers."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile
from PIL import Image
from phonetic_core.egg import EGGConfig, prepare, analyze_events
from phonetic_core.egg.export_series import spectral_series, waveform_series
from ptb_api.egg_models import EggTaskConfig
from ptb_worker.egg_exports import csv_bytes
from ptb_worker.egg_child import prepare as export
from ptb_worker.segmentation import unpack_bundle

FIXTURES = Path(__file__).resolve().parents[2]/'tests/fixtures/m03'


@pytest.fixture(scope='module')
def frozen():
    meta = json.loads((FIXTURES/'EGG-SYN-PCM16.json').read_text('utf-8'))
    with np.load(FIXTURES/'EGG-SYN-PCM16.npz',allow_pickle=False) as data: arrays=dict(data)
    samples = np.column_stack([arrays['load.egg_signal_raw'],arrays['load.audio_signal']])
    stream=io.BytesIO();wavfile.write(stream,44100,samples)
    return meta,arrays,samples,stream.getvalue()


@pytest.mark.parametrize('label,praat,gci,threshold', [
    ('batch',True,True,.01),('no_praat',False,True,.01),('no_gci_f0',True,False,.01),
    ('no_f0',False,False,.01),('masked',True,True,1.0),
])
def test_legacy_batch_csv_bytes_all_option_combinations(frozen,label,praat,gci,threshold):
    meta,arrays,samples,_=frozen
    cfg=EGGConfig.for_workbench(); result=analyze_events(prepare(samples,44100,cfg),cfg)
    result.audio_f0_times=arrays['praat.legacy.times'];result.audio_f0_values=arrays['praat.legacy.values']
    settings=EggTaskConfig(mode='batch',keep_praat_f0=praat,keep_gci_f0=gci,silence_threshold=threshold)
    raw,_=csv_bytes(result,cfg,settings,0,.8)
    assert hashlib.sha256(raw).hexdigest()==meta['exports'][label]['csv_sha256']


@pytest.mark.parametrize('window',[5,20,50])
def test_spectral_psd_matches_frozen_v2_image_array(frozen,window):
    _,arrays,_,_=frozen
    power,_,_=spectral_series(arrays['load.audio_signal'][4410:22050],44100,window)
    np.testing.assert_array_equal(np.flipud(10*np.log10(power)),arrays[f'spectrum.{window}.0'])


def test_corrected_waveform_time_and_retained_local_filter(frozen):
    _,arrays,samples,_=frozen;cfg=EGGConfig.for_workbench();result=prepare(samples,44100,cfg)
    times,audio,_=waveform_series(result,cfg,4410,22050)
    np.testing.assert_array_equal(times,np.arange(4410,22050)/44100)
    np.testing.assert_array_equal(audio,arrays['load.audio_signal'][4410:22050])
    assert times[-1] == (22050-1)/44100 < .5


@pytest.mark.parametrize('window',[5,20,50])
def test_blocked_spectral_fft_is_exactly_the_original_full_call(frozen,window):
    from matplotlib.mlab import specgram
    _,arrays,_,_=frozen;audio=np.tile(arrays['load.audio_signal'],4)
    nfft=int(44100*window/1000)
    old=specgram(audio,NFFT=nfft,Fs=44100,noverlap=int(nfft*.75))
    actual=spectral_series(audio,44100,window)
    for a,b in zip(actual,old):
        assert a.dtype==b.dtype and a.tobytes()==b.tobytes()


@pytest.mark.parametrize('mode,images,count,size', [('single',False,5,(1500,900)),('batch',True,5,(1200,600)),('batch',False,2,None)])
def test_real_complete_export_and_corrected_time(frozen,mode,images,count,size):
    _,arrays,_,raw=frozen
    config=dict(mode=mode,generate_images=images)
    if mode=='single':config.update(roi_start=.1,roi_end=.5)
    bundle=unpack_bundle(export(raw,config),64_000_000)
    blobs={f['name']:b for f,b in zip(bundle.manifest['files'],bundle.payloads)}
    assert len(blobs)==count
    metadata=json.loads(blobs['egg.ptb.json']);frame=pd.read_csv(io.BytesIO(blobs['egg_DATA.csv']),float_precision='round_trip')
    assert metadata['runtime']['blas']=='MKL 2025.3.0' and metadata['praat_time']=='pitch.xs()'
    if mode=='single':
        assert frame['Time (s)'].between(.1,.5,inclusive='left').all()
        times=arrays['praat.actual.times']; values=arrays['praat.actual.values']; take=(times>=.1)&(times<.5)&np.isfinite(values)
        actual=frame.dropna(subset=['F0_Praat (Hz)'])
        np.testing.assert_array_equal(actual['Time (s)'],times[take])
        np.testing.assert_array_equal(actual['F0_Praat (Hz)'],values[take])
    for name,blob in blobs.items():
        if name.endswith('.png'):
            with Image.open(io.BytesIO(blob)) as image: assert image.size==size;image.verify()


@pytest.mark.parametrize('order,key',[(None,'auto'),(12,'explicit')])
def test_inverse_pair_float64_exact_samples(frozen,order,key):
    _,arrays,_,raw=frozen
    bundle=unpack_bundle(export(raw,dict(mode='inverse',roi_end=.12,lp_order=order)),64_000_000)
    blobs={f['name']:b for f,b in zip(bundle.manifest['files'],bundle.payloads)}
    for name,expected in [('egg_ORIG.wav',arrays['load.audio_signal'][:5292]),('egg_IF.wav',arrays['inverse.'+key])]:
        rate,samples=wavfile.read(io.BytesIO(blobs[name]))
        assert rate==44100 and len(samples)==5292 and samples.dtype==np.float64
        np.testing.assert_array_equal(samples,expected.astype(np.float64))


@pytest.mark.parametrize('frames,channels,rate,config,code', [
    (1000,1,44100,{},'egg_stereo_required'),(1000,3,44100,{},'egg_stereo_required'),
    (8001,2,8000,{'roi_end':2.0},'egg_invalid_roi'),
    (96000,2,96000,{'mode':'inverse','roi_end':1.0},'egg_inverse_budget'),
    (1000,2,4000,{},'egg_sample_rate'),
    (480001,2,8000,{},'egg_input_budget'),
])
def test_child_rejects_unsupported_shape_and_budget_before_analysis(frames,channels,rate,config,code):
    from ptb_worker.acoustic_errors import AcousticFailure
    stream=io.BytesIO();wavfile.write(stream,rate,np.zeros((frames,channels),dtype=np.int16))
    with pytest.raises(AcousticFailure,match=code):export(stream.getvalue(),config)
