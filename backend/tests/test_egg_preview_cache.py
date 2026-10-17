"""Display-only changes reuse real EGG series without changing output bytes."""
import numpy as np
import pytest

from phonetic_core.egg import EGGConfig, prepare, analyze_events
from ptb_api.egg_models import EggTaskConfig
from ptb_worker import egg_preview


def test_single_linux_rpc_auto_db_and_audio_reuse_preserve_science(monkeypatch):
    import io,json
    from types import SimpleNamespace
    from scipy.io import wavfile
    from ptb_worker import egg_interactive_child as child,egg_runtime
    monkeypatch.setattr(egg_runtime,'fingerprint',lambda:None)
    fs=16000;t=np.arange(fs)/fs;stream=io.BytesIO();wavfile.write(stream,fs,np.column_stack([np.sin(2*np.pi*120*t),np.sin(2*np.pi*120*t+.3)]).astype('float32'));raw=stream.getvalue()
    config=EggTaskConfig(mode='preview',roi_start=.1,roi_end=.4,keep_praat_f0=False,keep_gci_f0=False).model_dump()
    def once(audio_ready=False,auto_db=False):
        source=io.BytesIO(json.dumps(dict(size=len(raw),config=config,audio_ready=audio_ready,auto_db=auto_db)).encode()+b'\n'+raw);target=io.BytesIO()
        monkeypatch.setattr(child,'sys',SimpleNamespace(stdin=SimpleNamespace(buffer=source),stdout=SimpleNamespace(buffer=target)))
        child.once();return json.loads(target.getvalue())
    first=once();reused=once(audio_ready=True)
    assert first['audio_base64'] and reused['audio_base64'] is None
    assert {k:v for k,v in first.items() if k!='audio_base64'}=={k:v for k,v in reused.items() if k!='audio_base64'}
    automatic=once(auto_db=True);session=child.Session(raw);original=session.update(config);lo,hi=original['preview']['suggested_db_range'];expected=session.update({**config,'spec_vmin':lo,'spec_vmax':hi});expected['audio_base64']=original['audio_base64']
    assert automatic==json.loads(json.dumps(expected)) and automatic['audio_base64']


@pytest.fixture
def signal():
    fs = 16000
    time = np.arange(fs)/fs
    samples = np.column_stack((.6*np.sin(2*np.pi*120*time), .5*np.sin(2*np.pi*120*time+.3)))
    settings = EggTaskConfig(mode='preview', roi_start=.1, roi_end=.4, micro_center=.2,
                             keep_praat_f0=False, keep_reaper_f0=False)
    config = EGGConfig.for_workbench()
    return analyze_events(prepare(samples, fs, config), config), config, settings


def test_db_and_micro_changes_reuse_science_and_match_uncached_bytes(signal, monkeypatch):
    result, config, settings = signal
    calls = dict(spectrum=0, cq=0)
    spectrum, cq = egg_preview.spectral_series, egg_preview.cq_segment

    def counted_spectrum(*args, **kwargs):
        calls['spectrum'] += 1
        return spectrum(*args, **kwargs)

    def counted_cq(*args, **kwargs):
        calls['cq'] += 1
        return cq(*args, **kwargs)

    monkeypatch.setattr(egg_preview, 'spectral_series', counted_spectrum)
    monkeypatch.setattr(egg_preview, 'cq_segment', counted_cq)
    cache = {}
    first, last = int(.1*result.fs), int(.4*result.fs)
    initial = egg_preview.preview_files(result, config, settings, first, last, cache=cache, include_audio=False)
    adjusted = settings.model_copy(update=dict(spec_vmin=-75., spec_vmax=-25.))
    cached = egg_preview.preview_files(result, config, adjusted, first, last, cache=cache, include_audio=False)
    assert calls == dict(spectrum=1, cq=1)
    assert initial[0]['cq'] == cached[0]['cq']
    assert initial[1]['egg_PSD.png'] != cached[1]['egg_PSD.png']
    assert cached == egg_preview.preview_files(result, config, adjusted, first, last, include_audio=False)
    before = calls.copy()
    moved = adjusted.model_copy(update=dict(micro_center=.25, micro_width_ms=30.))
    micro = egg_preview.preview_files(result, config, moved, first, last, cache=cache, include_audio=False)
    assert calls == before
    assert micro == egg_preview.preview_files(result, config, moved, first, last, include_audio=False)
    assert len(cache) == 1


def test_actual_window_and_roi_changes_invalidate_science(signal, monkeypatch):
    result, config, settings = signal
    calls = []
    spectrum = egg_preview.spectral_series
    monkeypatch.setattr(egg_preview, 'spectral_series', lambda *args, **kwargs: (calls.append(True), spectrum(*args, **kwargs))[1])
    cache = {}
    first, last = int(.1*result.fs), int(.4*result.fs)
    egg_preview.preview_files(result, config, settings, first, last, cache=cache, include_audio=False)
    changed = settings.model_copy(update=dict(spec_window_ms=10.))
    changed_config = EGGConfig.for_workbench(spec_window_ms=10.)
    egg_preview.preview_files(result, changed_config, changed, first, last, cache=cache, include_audio=False)
    egg_preview.preview_files(result, changed_config, changed, first, last+160, cache=cache, include_audio=False)
    assert len(calls) == 3
    assert len(cache) == 1


def test_spectrum_cache_budget_retains_no_oversized_array(signal, monkeypatch):
    result, config, settings = signal
    monkeypatch.setattr(egg_preview, 'SPECTRAL_CACHE_BYTES', 0)
    cache = {}
    first, last = int(.1*result.fs), int(.4*result.fs)
    egg_preview.preview_files(result, config, settings, first, last, cache=cache, include_audio=False)
    assert next(iter(cache.values()))['_spectral_db'] is None
    changed = settings.model_copy(update=dict(spec_vmin=-75., spec_vmax=-25.))
    assert egg_preview.preview_files(result, config, changed, first, last, cache=cache, include_audio=False) == egg_preview.preview_files(result, config, changed, first, last, include_audio=False)
