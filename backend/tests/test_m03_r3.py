"""R3 display additions, independently checked against numerical inputs."""
import io
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from phonetic_core.egg import EGGConfig, prepare as load
from phonetic_core.egg.export_series import spectral_series
from ptb_api.egg_models import EggInverseData, EggPreviewData
from ptb_worker.egg_child import prepare
from ptb_worker.segmentation import unpack_bundle
from ptb_worker.egg_exports import f0_axis_range


def test_export_f0_axis_has_no_50_500_clip_and_ignores_unvoiced():
    assert f0_axis_range([0,np.nan,np.inf,-1]) is None
    assert f0_axis_range([35,45]) == (30,50)
    assert f0_axis_range([600]) == (588,612)
    assert f0_axis_range([35,45,600,900]) == (0,987)


def fixture():
    path=Path(__file__).resolve().parents[2]/'tests/fixtures/m03/EGG-SYN-PCM16.npz'
    with np.load(path) as a:
        samples=np.column_stack([a['load.egg_signal_raw'],a['load.audio_signal']])
    raw=io.BytesIO();wavfile.write(raw,44100,samples)
    return samples,raw.getvalue()


def metadata(raw,config):
    bundle=unpack_bundle(prepare(raw,config),64_000_000)
    return json.loads(bundle.payloads[0])


def test_auto_db_uses_same_psd_only_visible_band_and_preserves_manual_range():
    samples,raw=fixture()
    config=dict(mode='preview',roi_start=.1,roi_end=.5,micro_center=.3,
                keep_praat_f0=False,keep_gci_f0=False,spec_vmin=-130,spec_vmax=-50)
    meta=metadata(raw,config)
    result=load(samples,44100,EGGConfig.for_workbench())
    power,freq,_=spectral_series(result.audio_signal[4410:22050],44100,20)
    db=10*np.log10(power[freq<=5000])
    upper=float(np.ceil(db[np.isfinite(db)].max()))
    assert meta['preview']['suggested_db_range']==[upper-50,upper]
    assert meta['config']['spec_vmin']==-130 and meta['config']['spec_vmax']==-50


def test_silence_has_no_invented_auto_db_suggestion():
    raw=io.BytesIO();wavfile.write(raw,8000,np.zeros((4000,2),np.int16))
    meta=metadata(raw.getvalue(),dict(mode='preview',roi_end=.5,keep_praat_f0=False,keep_gci_f0=False))
    assert meta['preview']['suggested_db_range'] is None


def test_inverse_full_egg_is_exact_same_task_selected_filtered_samples():
    samples,raw=fixture()
    meta=metadata(raw,dict(mode='inverse',roi_start=.1,roi_end=.5,lowpass_cutoff=2000))
    result=load(samples,44100,EGGConfig.for_workbench(lowpass_cutoff=2000))
    data=EggInverseData.model_validate(meta['inverse_view'])
    assert data.sample_rate_hz==44100
    np.testing.assert_array_equal(data.full_egg_values,result.egg_signal_processed[4410:22050])
    assert len(data.full_egg_values)>len(data.egg_values)
    old={k:v for k,v in meta['inverse_view'].items() if k not in ('full_egg_values','sample_rate_hz')}
    assert EggInverseData.model_validate(old).full_egg_values is None


def test_old_preview_without_display_suggestion_still_loads():
    _,raw=fixture()
    data=metadata(raw,dict(mode='preview',roi_end=.5,keep_praat_f0=False,keep_gci_f0=False))['preview']
    data.pop('suggested_db_range')
    assert EggPreviewData.model_validate_json(json.dumps(data)).suggested_db_range is None
