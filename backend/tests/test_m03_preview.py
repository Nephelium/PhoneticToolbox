"""Preview serialization against independent V2 frozen arrays and file readers."""
import io
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from PIL import Image
from ptb_worker.egg_child import prepare
from ptb_worker.segmentation import unpack_bundle
from ptb_api.egg_models import EggTaskConfig, EggPreviewData

FIXTURES=Path(__file__).resolve().parents[2]/'tests/fixtures/m03'


def test_preview_without_gci_f0_only_detects_events_in_display_windows(monkeypatch):
    import phonetic_core.egg as egg
    def unexpected(*args, **kwargs):
        raise AssertionError('A local preview without GCI F0 must not redetect full-file events')
    monkeypatch.setattr(egg, 'analyze_events', unexpected)
    with np.load(FIXTURES/'EGG-SYN-PCM16.npz') as a:
        samples=np.column_stack([a['load.egg_signal_raw'],a['load.audio_signal']])
    stream=io.BytesIO();wavfile.write(stream,44100,samples)
    bundle=unpack_bundle(prepare(stream.getvalue(),dict(mode='preview',roi_start=.1,roi_end=.5,
        micro_center=.3,keep_praat_f0=False,keep_gci_f0=False)),64_000_000)
    preview=json.loads(bundle.payloads[0])['preview']
    assert preview['cq']['times'] and preview['gci'] and preview['goi']
    assert preview['gci_f0']['times']==[]

@pytest.mark.parametrize('raw_mode',[False,True])
def test_preview_exact_audio_praat_and_spectrum_extent(raw_mode):
    with np.load(FIXTURES/'EGG-SYN-PCM16.npz') as a:
        samples=np.column_stack([a['load.egg_signal_raw'],a['load.audio_signal']])
        stream=io.BytesIO();wavfile.write(stream,44100,samples)
        bundle=unpack_bundle(prepare(stream.getvalue(),dict(mode='preview',roi_start=.1,roi_end=.5,micro_center=.3,signal_mode='raw' if raw_mode else 'filtered')),64_000_000)
        files={f['name']:v for f,v in zip(bundle.manifest['files'],bundle.payloads)}
        assert set(files)=={'egg.ptb.json','egg_AUDIO.wav','egg_PSD.png'}
        meta=json.loads(files['egg.ptb.json']);p=EggPreviewData.model_validate_json(json.dumps(meta['preview']))
        rate,audio=wavfile.read(io.BytesIO(files['egg_AUDIO.wav']))
        assert rate==44100 and audio.dtype==np.float32
        np.testing.assert_array_equal(audio,a['load.audio_signal'])
        times=np.arange(int((.3-.025)*44100),int((.3+.025)*44100))/44100
        np.testing.assert_array_equal(p.audio.times,times)
        np.testing.assert_array_equal(p.audio.values,a['load.audio_signal'][int((.3-.025)*44100):int((.3+.025)*44100)])
        if raw_mode:np.testing.assert_array_equal(p.egg.values,a['load.egg_signal_raw'][int((.3-.025)*44100):int((.3+.025)*44100)])
        pt=a['praat.actual.times'];take=(pt>=.1)&(pt<.5)
        np.testing.assert_array_equal(p.praat.times,pt[take])
        np.testing.assert_array_equal(np.array(p.praat.values,dtype=float),a['praat.actual.values'][take])
        assert p.spectral_extent[:2]==(.1,.5)
        assert len(p.cq.times)==len(p.sq.values)>0
        with Image.open(io.BytesIO(files['egg_PSD.png'])) as im:
            assert im.width<=1024 and im.height<=512 and im.mode=='L';im.verify()

def test_silent_preview_contains_nulls_or_empty_series_not_fake_pitch():
    stream=io.BytesIO();wavfile.write(stream,8000,np.zeros((4000,2),np.int16))
    bundle=unpack_bundle(prepare(stream.getvalue(),dict(mode='preview',roi_end=.5)),64_000_000)
    meta=json.loads(bundle.payloads[0]);p=meta['preview']
    assert not p['cq']['times'] and not p['gci'] and not p['goi']
    assert all(v is None for v in p['praat']['values'])
    assert all(v==0 for v in p['audio']['values'])

@pytest.mark.parametrize('config',[dict(mode='single',micro_center=.1),dict(mode='preview',micro_width_ms=4.99),dict(mode='preview',micro_width_ms=5000.01)])
def test_micro_bounds_are_explicit(config):
    with pytest.raises(ValueError):EggTaskConfig(**config)


@pytest.mark.parametrize('label,center,width',[('middle',.3,50),('start',0.,10),('end',.8,200)])
@pytest.mark.parametrize('raw',[False,True])
def test_micro_display_matches_original_qt_plot_samples(label,center,width,raw):
    from phonetic_core.egg import EGGConfig,prepare as load
    from phonetic_core.egg.preview import micro_waveforms
    with np.load(FIXTURES/'EGG-SYN-PCM16.npz') as a,np.load(FIXTURES/'ui-display.npz') as old:
        config=EGGConfig.for_workbench();r=load(np.column_stack([a['load.egg_signal_raw'],a['load.audio_signal']]),44100,config)
        times,_,values=micro_waveforms(r,config,center,width,raw=raw)
        np.testing.assert_array_equal(values,old[f'{label}.{raw}.egg'])
        np.testing.assert_array_equal((times-center)*1000,old[f'{label}.{raw}.ms'])


def test_inverse_comparison_matches_original_dialog_all_four_plots():
    from phonetic_core.egg import EGGConfig,prepare as load
    from phonetic_core.egg.preview import inverse_comparison
    with np.load(FIXTURES/'EGG-SYN-PCM16.npz') as a,np.load(FIXTURES/'ui-display.npz') as old:
        config=EGGConfig.for_workbench();r=load(np.column_stack([a['load.egg_signal_raw'],a['load.audio_signal']]),44100,config)
        frequencies,spectra,times,waves=inverse_comparison(r.audio_signal[:5292],a['inverse.auto'],r.egg_signal_processed[:5292],44100)
        for values,key in zip(spectra,['if.0.0','if.0.1','if.3.0']):
            np.testing.assert_array_equal(frequencies,old[key+'.x']);np.testing.assert_array_equal(values,old[key+'.y'])
        for values,key in zip(waves,['if.1.0','if.2.0','if.4.0']):
            np.testing.assert_array_equal(times*1000,old[key+'.x']);np.testing.assert_array_equal(values,old[key+'.y'])
