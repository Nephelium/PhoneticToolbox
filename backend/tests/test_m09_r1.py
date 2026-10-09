import io
import json
import base64
import cv2
import numpy as np
import soundfile as sf
import pytest
from ptb_api.spec2wav_models import Spec2WavConfig
from ptb_worker.spec2wav_child import prepare,preview
from ptb_worker.segmentation import unpack_bundle


def test_audio_preview_and_result_share_real_grid_and_float_wav():
    audio=np.random.default_rng(4).normal(0,.05,(8001,2));raw=io.BytesIO();sf.write(raw,audio,16000,format='WAV',subtype='FLOAT')
    config=dict(mode='audio_draw');view=preview(raw.getvalue(),config)
    bundle=unpack_bundle(prepare(raw.getvalue(),config),64_000_000)
    target=cv2.imdecode(np.frombuffer(bundle.payloads[0],np.uint8),0)
    baseline=cv2.imdecode(np.frombuffer(base64.b64decode(view['image_base64']),np.uint8),0)
    np.testing.assert_array_equal(target,baseline)
    decoded,sr=sf.read(io.BytesIO(bundle.payloads[2]));original,_=sf.read(io.BytesIO(raw.getvalue()))
    np.testing.assert_allclose(decoded,original,atol=1e-12,rtol=0)
    assert sr==16000 and decoded.shape==(8001,2)
    meta=json.loads(bundle.payloads[3]);assert meta['n_iter']==0 and meta['phase_method']=='original-stft-phase'


def test_image_draw_calibrates_before_painting_and_uses_griffin_lim():
    gray=np.full((81,121),255,np.uint8);gray[:,30:33]=0
    raw=cv2.imencode('.png',gray)[1].tobytes()
    config=dict(mode='image_draw',time_end=.1,n_iter=2,corners=[dict(x=0,y=0),dict(x=.8,y=.2),dict(x=1,y=.9),dict(x=.1,y=1)],strokes=[dict(color=0,size=7,opacity=.5,points=[dict(x=.5,y=.5)])])
    view=preview(raw,config);bundle=unpack_bundle(prepare(raw,config),64_000_000)
    target=cv2.imdecode(np.frombuffer(bundle.payloads[0],np.uint8),0)
    assert target.shape==(view['height'],view['width'])
    meta=json.loads(bundle.payloads[3]);assert meta['phase_method']=='griffin-lim' and meta['geometry']=='perspective-four-corners'
    assert meta['config']['corners']==config['corners'] and meta['stroke_count']==1


@pytest.mark.parametrize('config',[dict(mode='audio_draw',corners=[dict(x=0,y=0)]*4),dict(strokes=[dict(points=[dict(x=.5,y=.5)])]),dict(mode='audio_draw',strokes=[dict(opacity=1.01,points=[dict(x=.5,y=.5)])]),dict(mode='audio_draw',strokes=[dict(points=[dict(x=2,y=.5)])]),dict(mode='audio_draw',channel=2)])
def test_wire_bounds(config):
    with pytest.raises(ValueError):Spec2WavConfig(**config)
