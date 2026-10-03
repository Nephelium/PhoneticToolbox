"""M03-R5: independent analytic tones, native engine, old-request compatibility."""
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile
from scipy.signal import sawtooth
from phonetic_core.egg.f0 import praat_pitch,reaper_pitch
from ptb_api.egg_models import EggTaskConfig
from ptb_worker.egg_child import prepare
from ptb_worker.egg_interactive_child import Session
from ptb_worker.egg_f0 import bounds,NATIVE_BYTES
from ptb_worker.io.scratch import Scratch
from ptb_worker.io.limits import Limits
from ptb_worker.native.reaper import Reaper,REAPER_SHA256
from ptb_worker.segmentation import unpack_bundle

ROOT=Path(__file__).resolve().parents[2]
RATE=16000


def tone(frequency,seconds=1):
    t=np.arange(round(RATE*seconds))/RATE
    return .6*sawtooth(2*np.pi*frequency*t,width=.15)


def wav(audio):
    b=io.BytesIO();wavfile.write(b,RATE,np.column_stack([tone(120,len(audio)/RATE),audio]));return b.getvalue()


@pytest.fixture
def native(tmp_path):
    with Scratch(tmp_path,NATIVE_BYTES) as scratch:
        yield Reaper(ROOT/'phonetic_toolbox/core/acoustic/reaper.exe',scratch,
            Limits(input_bytes=NATIVE_BYTES,samples=5_760_000,output_bytes=2_000_000))


@pytest.mark.parametrize('frequency',[40,700])
def test_real_low_and_high_tones_extend_previous_search(frequency,native):
    audio=tone(frequency)
    for track in [praat_pitch(audio,RATE,pitch_floor=30.,pitch_ceiling=800.),reaper_pitch(audio,RATE,native)]:
        voiced=track.values[(track.times>.15)&(track.times<.85)]
        assert np.isfinite(voiced).sum()>30
        assert abs(np.nanmedian(voiced)-frequency)/frequency < .03


def test_native_time_grid_and_values_are_not_shifted_or_interpolated(native):
    from phonetic_core.models.audio import AudioInput
    audio=tone(180)
    direct=native(AudioInput(audio,RATE),.01,30.,800.,hilbert=False,no_highpass=False)
    actual=reaper_pitch(audio,RATE,native)
    np.testing.assert_array_equal(actual.times,direct.times)
    np.testing.assert_array_equal(actual.values,direct.values)


@pytest.mark.parametrize('seconds',[.05,.5])
def test_silent_and_short_tracks_do_not_fabricate_f0(seconds,native):
    audio=np.zeros(round(seconds*RATE))
    for track in [praat_pitch(audio,RATE,pitch_floor=30.,pitch_ceiling=800.),reaper_pitch(audio,RATE,native)]:
        assert not np.isfinite(track.values).any()


def test_legacy_bounds_remain_explicit():
    assert bounds(EggTaskConfig())==(75.,600.)
    assert bounds(EggTaskConfig(f0_policy='audio-f0/2'))==(30.,800.)
    with pytest.raises(ValueError):EggTaskConfig(keep_reaper_f0=True)


def test_preview_caches_reaper_and_recovers_from_unavailable(native):
    raw=wav(tone(180));s=Session(raw)
    c=dict(mode='preview',roi_end=.5,f0_policy='audio-f0/2',keep_reaper_f0=True,keep_gci_f0=False)
    with pytest.raises(ValueError,match='egg_reaper_unavailable'):s.update(c)
    assert s.update(c|dict(keep_reaper_f0=False))['preview']['reaper'] is None
    class Count:
        count=0
        def __call__(self,*args,**kwargs):self.count+=1;return native(*args,**kwargs)
    counter=Count();s.reaper=counter
    first=s.update(c)['preview'];assert first['reaper']['times']
    assert first['reaper']['values']!=first['praat']['values']
    s.update(c|dict(micro_center=.3));s.update(c|dict(keep_reaper_f0=False));s.update(c)
    assert counter.count==1
    s.update(c|dict(flip_channels=True));assert counter.count==2


@pytest.mark.parametrize('mode',['single','batch'])
def test_native_reaper_export_metadata_csv_and_png(mode,native):
    c=dict(mode=mode,f0_policy='audio-f0/2',keep_reaper_f0=True,generate_images=True)
    if mode=='single':c.update(roi_start=.1,roi_end=.7)
    bundle=unpack_bundle(prepare(wav(tone(180)),c,reaper=native),64_000_000)
    files=dict(zip([f['name'] for f in bundle.manifest['files']],bundle.payloads))
    frame=pd.read_csv(io.BytesIO(files['egg_DATA.csv']))
    assert frame['F0_REAPER (Hz)'].notna().any()
    assert abs(frame['F0_REAPER (Hz)'].median()-180)<5
    if mode=='single':assert frame['Time (s)'].min()>=.1 and frame['Time (s)'].max()<.7
    meta=json.loads(files['egg.ptb.json']);assert meta['f0_analysis']['revision']=='audio-f0/2'
    assert meta['f0_analysis']['praat']['floor_hz']==30
    assert meta['f0_analysis']['praat']['ceiling_hz']==800
    assert meta['f0_analysis']['reaper']['binary_sha256']==REAPER_SHA256
    assert 'SRC-REAPER' in meta['source_ids']
    assert files['egg_SPEC_F0.png'].startswith(b'\x89PNG')


def test_new_f0_policy_leaves_egg_events_and_psd_unchanged(native):
    raw=wav(tone(180));s=Session(raw,reaper=native)
    old=s.update(dict(mode='preview',roi_end=.5,keep_praat_f0=True))
    new=s.update(dict(mode='preview',roi_end=.5,keep_praat_f0=True,f0_policy='audio-f0/2',keep_reaper_f0=True))
    for key in ['cq','sq','gci_f0','audio','egg','gci','goi']:
        assert old['preview'][key]==new['preview'][key]
    assert old['psd_base64']==new['psd_base64']
