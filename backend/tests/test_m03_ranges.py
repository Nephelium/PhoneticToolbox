"""A19: original V2 display arrays at expanded endpoints, no generated expectations."""
import io
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from ptb_api.egg_models import EggTaskConfig
from ptb_worker.egg_child import prepare
from ptb_worker.segmentation import unpack_bundle

FIXTURES = Path(__file__).resolve().parents[2]/'tests/fixtures/m03'
CASES = [('min',3.2,5),('max',3.2,5000),('start',0.,5),('end',6.4,5),('wide_start',0.,5000),('wide_end',6.4,5000)]


@pytest.mark.parametrize('width',[5,5000])
def test_restore_original_wheel_endpoints(width):
    assert EggTaskConfig(mode='preview',micro_width_ms=width).micro_width_ms==width


@pytest.mark.parametrize('label,center,width',CASES)
@pytest.mark.parametrize('raw',[False,True])
def test_actual_preview_matches_original_display(label,center,width,raw):
    with np.load(FIXTURES/'EGG-SYN-PCM16.npz') as a:
        samples=np.tile(np.column_stack([a['load.egg_signal_raw'],a['load.audio_signal']]),(8,1))
    stream=io.BytesIO();wavfile.write(stream,44100,samples)
    bundle=unpack_bundle(prepare(stream.getvalue(),dict(mode='preview',roi_start=3.,roi_end=3.5,
        micro_center=center,micro_width_ms=width,signal_mode='raw' if raw else 'filtered',keep_praat_f0=False)),64_000_000)
    meta=json.loads(bundle.payloads[0]);data=meta['preview']
    evidence=json.loads((FIXTURES/'ranges-source.json').read_text('utf-8'))
    assert meta['input_sha256']==evidence['input_sha256']
    assert data['micro_sample_stride']==evidence['strides'][f'{label}.{raw}']
    with np.load(FIXTURES/'ranges.npz') as old:
        for name in ('audio','egg'):
            np.testing.assert_array_equal(data[name]['values'],old[f'{label}.{raw}.{name}'])
            np.testing.assert_array_equal((np.array(data[name]['times'])-center)*1000,old[f'{label}.{raw}.{name}_ms'])
            assert len(data[name]['values'])<=10001


@pytest.mark.parametrize('fs',[8000,48000,96000])
def test_wide_high_pitch_keeps_all_events(fs):
    t=np.arange(fs*6)/fs
    samples=np.column_stack([np.sin(2*np.pi*300*t),np.sin(2*np.pi*300*t)]).astype(np.float32)
    stream=io.BytesIO();wavfile.write(stream,fs,samples)
    bundle=unpack_bundle(prepare(stream.getvalue(),dict(mode='preview',roi_start=2.,roi_end=2.5,
        micro_center=3.,micro_width_ms=5000,keep_praat_f0=False)),64_000_000)
    data=json.loads(bundle.payloads[0])['preview']
    assert 1400<len(data['gci'])<1600
    assert data['micro_sample_stride']>1
    assert len(data['audio']['times'])<=10001
