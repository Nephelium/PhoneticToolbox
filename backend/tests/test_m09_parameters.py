"""Parameter preview works with no component installation or Torch import."""
import sys
import numpy as np
import pytest
from ptb_worker.m09_parameters import predict, weights
from ptb_worker.spec2wav_child import preview


@pytest.mark.parametrize('algorithm', ['griffin_lim', 'neural_pipeline'])
@pytest.mark.parametrize('axis,lower', [('linear', 0), ('log', 50)])
def test_builtin_preview_without_download(tmp_path, monkeypatch, axis, lower, algorithm):
    import cv2
    from ptb_worker import vocoder_components
    monkeypatch.setenv('PTB_VOCODER_ROOT', str(tmp_path / 'absent'))
    def unexpected(*args, **kwargs):
        raise AssertionError('Prediction must not inspect or download optional components')
    monkeypatch.setattr(vocoder_components, 'installed', unexpected)
    monkeypatch.setattr(vocoder_components, 'install', unexpected)
    gray = np.full((48, 96), 220, np.uint8)
    gray[8:12, ::6] = 20
    raw = cv2.imencode('.png', gray)[1].tobytes()
    result = preview(raw, dict(algorithm=algorithm, auto_parameters=True, frequency_axis=axis,
                              time_end=1, freq_start=lower, freq_end=8000))
    p = result['prediction']
    assert p['win_length_ms'] == int(p['win_length_ms'])
    assert p['hop_ms'] == p['win_length_ms'] / 4
    assert p['min_db'] < p['max_db'] == 0
    assert not p['absolute_db_predicted']
    assert 'torch' not in sys.modules
    assert not (tmp_path / 'absent').exists()


def test_prediction_is_repeatable_and_does_not_mutate_input():
    x = np.random.default_rng(11).integers(0, 256, (64, 90), dtype=np.uint8)
    before = x.copy()
    config = dict(time_start=0, time_end=1, freq_start=0, freq_end=8000)
    assert predict(x, config) == predict(x, config)
    assert np.array_equal(x, before)
    assert all(not v.flags.writeable for v in weights().values())


@pytest.mark.parametrize('override', [dict(time_end=0), dict(freq_end=0),
    dict(frequency_axis='log'), dict(frequency_axis='unknown'), dict(time_end=float('nan'))])
def test_invalid_calibration_is_rejected(override):
    with pytest.raises(ValueError, match='invalid_spectrogram_config'):
        predict(np.zeros((48, 96), np.uint8), dict(time_start=0, time_end=1, freq_start=0, freq_end=8000) | override)


def test_manual_preview_skips_model(monkeypatch):
    import cv2
    import ptb_worker.m09_parameters as parameters
    def unexpected(*args, **kwargs):raise AssertionError('Manual preview must skip prediction')
    monkeypatch.setattr(parameters, 'predict', unexpected)
    raw = cv2.imencode('.png', np.zeros((8, 8), np.uint8))[1].tobytes()
    assert 'prediction' not in preview(raw, dict(algorithm='neural_pipeline', auto_parameters=False))


def test_auto_gl_renders_with_builtin_parameters_without_optional_component(tmp_path, monkeypatch):
    import cv2
    import json
    from ptb_worker.spec2wav_child import prepare
    from ptb_worker.segmentation import unpack_bundle
    monkeypatch.setenv('PTB_VOCODER_ROOT',str(tmp_path/'absent'))
    raw=cv2.imencode('.png',np.full((48,96),220,np.uint8))[1].tobytes()
    config=dict(auto_parameters=True,time_end=.2,freq_end=8000,n_iter=2)
    predicted=preview(raw,config)['prediction']
    bundle=unpack_bundle(prepare(raw,config),64_000_000)
    meta=json.loads(bundle.payloads[3])
    assert meta['prediction']==predicted
    assert meta['phase_method']=='griffin-lim' and meta['parameter_backend']=='builtin-numpy-float32'
    assert meta['hop_length']*4==meta['win_length']
    assert 'vocoder' not in meta
    assert not (tmp_path/'absent').exists()


def test_old_snapshots_keep_algorithm_specific_defaults():
    from ptb_api.spec2wav_models import Spec2WavConfig
    assert not Spec2WavConfig().auto_parameters
    assert Spec2WavConfig(algorithm='neural_pipeline').auto_parameters
    assert 'auto_parameters' not in Spec2WavConfig().snapshot()
    assert 'auto_parameters' not in Spec2WavConfig(algorithm='neural_pipeline').snapshot()
    assert Spec2WavConfig(auto_parameters=True).snapshot()['auto_parameters']
    assert not Spec2WavConfig(algorithm='neural_pipeline',auto_parameters=False).snapshot()['auto_parameters']


def test_manual_prediction_values_keep_same_gl_path_and_input_geometry():
    import cv2,json,io,soundfile as sf
    from ptb_worker.spec2wav_child import prepare
    from ptb_worker.segmentation import unpack_bundle
    gray=np.full((40,90),220,np.uint8);gray[12:17,::5]=10
    raw=cv2.imencode('.png',gray)[1].tobytes()
    config=dict(auto_parameters=True,reconstruction_profile='calibrated',time_end=.123,freq_end=5000,n_iter=2,target_sr=0)
    predicted=preview(raw,config)['prediction']
    manual=dict(config,auto_parameters=False,**{k:predicted[k] for k in ('win_length_ms','min_db','max_db')})
    automatic=unpack_bundle(prepare(raw,config),64_000_000)
    edited=unpack_bundle(prepare(raw,manual),64_000_000)
    assert automatic.payloads[2]==edited.payloads[2]
    meta=json.loads(edited.payloads[3]);audio,sr=sf.read(io.BytesIO(edited.payloads[2]))
    assert sr==10000 and len(audio)==1230
    assert meta['schema_version']=='m09-calibrated/1'
    assert cv2.imdecode(np.frombuffer(edited.payloads[1],np.uint8),0).shape==gray.shape
    assert meta['comparison_grid']['freq_end']==5000
