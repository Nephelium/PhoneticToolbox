"""M03-R8: shared adjustable search bounds, actual native calls and cache keys."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile
from scipy.signal import sawtooth

from phonetic_core.egg.f0 import praat_pitch, reaper_pitch
from ptb_api.egg_models import EggTaskConfig
from ptb_worker.egg_child import prepare
from ptb_worker.egg_f0 import bounds, NATIVE_BYTES
from ptb_worker.egg_interactive_child import Session
from ptb_worker.io.scratch import Scratch
from ptb_worker.io.limits import Limits
from ptb_worker.native.reaper import Reaper
from ptb_worker.segmentation import unpack_bundle
from scripts.verification_artifacts import verification_run

ROOT = Path(__file__).resolve().parents[2]
RATE = 16000


def samples(seconds=1):
    time = np.arange(round(RATE * seconds)) / RATE
    return np.column_stack([.6 * sawtooth(2 * np.pi * 120 * time, width=.15),
                            .6 * sawtooth(2 * np.pi * 180 * time, width=.15)])


def raw():
    buffer = io.BytesIO()
    wavfile.write(buffer, RATE, samples())
    return buffer.getvalue()


@pytest.fixture
def native(tmp_path):
    with Scratch(tmp_path, NATIVE_BYTES) as scratch:
        yield Reaper(ROOT / 'phonetic_toolbox/core/acoustic/reaper.exe', scratch,
                     Limits(input_bytes=NATIVE_BYTES, samples=5_760_000, output_bytes=2_000_000))


@pytest.mark.parametrize('changes', [dict(f0_min_hz=9), dict(f0_max_hz=2001),
                                    dict(f0_min_hz=300, f0_max_hz=200), dict(f0_min_hz=float('nan'))])
def test_contract_rejects_invalid_shared_bounds(changes):
    with pytest.raises(ValueError):
        EggTaskConfig(f0_policy='audio-f0/2', **changes)


def test_old_defaults_are_preserved():
    assert bounds(EggTaskConfig()) == (75., 600.)
    assert bounds(EggTaskConfig(f0_policy='audio-f0/2')) == (30., 800.)
    assert bounds(EggTaskConfig(f0_policy='audio-f0/2', f0_min_hz=80, f0_max_hz=260)) == (80, 260)


def test_custom_native_pitch_matches_direct_upstream_call(native):
    from phonetic_core.models.audio import AudioInput
    audio = samples()[:, 1]
    direct = native(AudioInput(audio, RATE), .01, 80., 260., hilbert=False, no_highpass=False)
    actual = reaper_pitch(audio, RATE, native, pitch_floor=80., pitch_ceiling=260.)
    np.testing.assert_array_equal(actual.times, direct.times)
    np.testing.assert_array_equal(actual.values, direct.values)
    assert abs(np.nanmedian(actual.values) - 180) < 5


def test_low_floor_short_praat_returns_no_complete_frames():
    actual = praat_pitch(samples(.2)[:, 1], RATE, pitch_floor=10., pitch_ceiling=260.)
    assert len(actual.times) == 0


@pytest.mark.parametrize('long', [False, True])
def test_range_edits_recompute_both_detectors_and_preserve_other_science(native, long):
    class Count:
        calls = []

        def __call__(self, audio, step, low, high, **kwargs):
            self.calls.append((low, high))
            return native(audio, step, low, high, **kwargs)

    counter = Count()
    counter.calls = []
    with verification_run('m03-r8', 'pitch-cache', f'Analytic 120/180 Hz stereo sawtooth; 16000 Hz; long={long}') as (_, scratch):
        if long:
            source = scratch / 'long.wav'
            wavfile.write(source, RATE, samples(121))
            session = Session(source_path=source, reaper=counter)
        else:
            session = Session(raw(), reaper=counter)
        try:
            config = dict(mode='preview', roi_start=.2, roi_end=.7, micro_center=.45,
                          f0_policy='audio-f0/2', keep_praat_f0=True, keep_reaper_f0=True)
            old = session.update(config)
            custom = session.update(config | dict(f0_min_hz=80., f0_max_hz=260.))
            assert counter.calls == [(30., 800.), (80., 260.)]
            assert custom['preview']['praat'] != old['preview']['praat']
            for key in ['cq', 'sq', 'gci_f0', 'audio', 'egg', 'gci', 'goi']:
                assert custom['preview'][key] == old['preview'][key]
            assert custom['psd_base64'] == old['psd_base64']
            again = session.update(config | dict(f0_min_hz=80., f0_max_hz=260., micro_center=.46))
            assert again['preview']['reaper'] == custom['preview']['reaper']
            assert len(counter.calls) == 2
            restored = session.update(config)
            assert restored['preview']['praat'] == old['preview']['praat']
            assert restored['preview']['reaper'] == old['preview']['reaper']
            assert len(counter.calls) == (2 if long else 3)
        finally:
            if session.recording:
                session.recording.close()


@pytest.mark.parametrize('mode', ['single', 'batch'])
def test_actual_exports_record_custom_bounds_for_both_detectors(mode, native):
    config = dict(mode=mode, f0_policy='audio-f0/2', keep_reaper_f0=True,
                  f0_min_hz=80., f0_max_hz=260.)
    bundle = unpack_bundle(prepare(raw(), config, reaper=native), 64_000_000)
    files = dict(zip([f['name'] for f in bundle.manifest['files']], bundle.payloads))
    metadata = json.loads(files['egg.ptb.json'])
    for key in ['praat', 'reaper']:
        assert metadata['f0_analysis'][key]['floor_hz'] == 80
        assert metadata['f0_analysis'][key]['ceiling_hz'] == 260
    frame = pd.read_csv(io.BytesIO(files['egg_DATA.csv']))
    for key in ['F0_Praat (Hz)', 'F0_REAPER (Hz)']:
        assert abs(frame[key].median() - 180) < 5
