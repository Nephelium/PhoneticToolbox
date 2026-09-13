"""M04 real MKL/Agg export readback; use Invoke-M03-Python.ps1."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from PIL import Image
from ptb_worker.lpc_child import prepare
from ptb_worker.segmentation import unpack_bundle
from ptb_worker.acoustic_errors import AcousticFailure


def wav(audio,rate=16000):
    stream=io.BytesIO();wavfile.write(stream,rate,audio);return stream.getvalue()


@pytest.mark.parametrize('case', ['default','dynamic_2k','noise_order200'])
def test_frozen_spectrum_png_and_wav(case):
    root=Path(__file__).resolve().parents[2]
    meta=json.loads((root/'tests/fixtures/m04/result.json').read_text('utf-8'))['cases'][case]
    with np.load(root/'tests/fixtures/m04/arrays.npz') as data:
        audio=data[case+'.input'];before=audio.tobytes()
        settings=dict(meta['config'])|dict(roi_end=len(audio)/meta['rate'])
        raw=wav(audio,meta['rate']);bundle=unpack_bundle(prepare(raw,settings,'ɑ̃˥ 测试.wav'),8_000_000)
        files=dict(zip([f['name'] for f in bundle.manifest['files']],bundle.payloads))
        result=json.loads(files['lpc.ptb.json']);actual=result['spectrum']
        for key,array in [('frequencies_hz','frequency'),('magnitude_db','db')]:
            assert np.asarray(actual[key]).tobytes()==data[case+'.'+array].tobytes()
        assert result['input_sha256']==hashlib.sha256(raw).hexdigest()
        assert result['render_fonts']['ipa']['family']=='Doulos SIL'
        image=Image.open(io.BytesIO(files['lpc_SPECTRUM.png']))
        assert image.size==(2400,1350) and abs(image.info['dpi'][0]-300)<.01
        assert image.convert('RGB').getpixel((0,0))==(255,255,255)
        rate,returned=wavfile.read(io.BytesIO(files['lpc_AUDIO.wav']))
        assert rate==meta['rate'] and returned.tobytes()==before
        assert audio.tobytes()==before


@pytest.mark.parametrize('raw,settings,code', [
    (b'bad',{'roi_end':.1},'invalid_audio'),
    (wav(np.ones(48001),48000),{'roi_end':48001/48000},'lpc_roi_budget'),
    (wav(np.ones(100),16000),{'roi_end':.1},'lpc_invalid_roi'),
    (wav(np.ones(100),1000),{'roi_end':.1},'lpc_sample_rate'),
    (wav(np.ones(1000)),{'roi_end':.01,'font':{'latin':'PTB Missing Font'}},'font_unavailable')],ids=['corrupt','roi_budget','roi_bounds','sample_rate','missing_font'])
def test_explicit_failures(raw,settings,code):
    with pytest.raises(AcousticFailure) as error:prepare(raw,settings)
    assert error.value.code==code


def test_name_safety_and_collision_context():
    from ptb_worker.lpc_exports import export_names
    names=export_names('CON.wav','ɑ̃˥/a:*末',.1,.2)
    assert all(len(n)<220 and not any(c in n for c in '/\\:<>"|?*') for n in names.values())
    assert 'ɑ̃˥' in names['lpc_SPECTRUM.png']
    assert names!=export_names('CON.wav','ɑ̃˥/a:*末',.2,.3)


@pytest.mark.parametrize('channels', [1,8])
def test_pcm_roi_preserves_conversion_and_sample_bounds(channels):
    source=np.arange(4000,dtype=np.int16)
    if channels>1:source=np.column_stack([source]*channels)
    bundle=unpack_bundle(prepare(wav(source),{'roi_start':.01001,'roi_end':.02001}),8_000_000)
    files=dict(zip([f['name'] for f in bundle.manifest['files']],bundle.payloads))
    rate,audio=wavfile.read(io.BytesIO(files['lpc_AUDIO.wav']))
    expected=np.arange(160,320,dtype=np.float64)/32768
    assert audio.tobytes()==expected.tobytes() and rate==16000


def test_invalid_grid_and_missing_layer():
    raw=wav(np.random.default_rng(4).normal(size=1600))
    with pytest.raises(AcousticFailure,match='invalid_textgrid'):
        prepare(raw,{'roi_end':.1},textgrid=b'broken grid')
    with pytest.raises(AcousticFailure,match='missing_or_invalid_tier'):
        prepare(raw,{'roi_end':.1,'tier_name':'missing'})


def test_more_than_eight_channels_rejected():
    with pytest.raises(AcousticFailure,match='lpc_input_budget'):
        prepare(wav(np.ones((1600,9))),{'roi_end':.1})
