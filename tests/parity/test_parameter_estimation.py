"""M01-B: independent P03/M01-A goldens; strict masks/time and original tolerance."""
import dataclasses
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from phonetic_core.models.audio import AudioInput
from phonetic_core.models.acoustic import AcousticConfig
from phonetic_core.ports.acoustic import AcousticBackends
from phonetic_core.services.acoustic import analyze_audio
from m01_science import ROOT, native_probe, associations, fail_irapt, pack


def load_support(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


support = load_support('baseline_support')
recipes = load_support('m01_baseline_support')
CASES = [c for c in recipes.cases() if c.get('mode', 'analysis') == 'analysis']


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_m01_independent_reference(case, tmp_path):
    old = support.load_json(ROOT/'tests/fixtures/m01'/f"{case['id']}.json.gz")['scientific']
    recipe = next(r for r in support.RECIPES if r['id'] == ('SYN-EMPTY' if case.get('empty') else 'SYN-VOWEL-44100'))
    path = support.create_fixture(tmp_path, recipe)
    fs, data = wavfile.read(path)
    before = data.copy()
    config = AcousticConfig(**{k: v for k, v in old['config'].items() if k != 'reaper_bin_path'})
    probe = native_probe(tmp_path, case.get('fault') == 'reaper_invalid_argument')
    result = analyze_audio(AudioInput(data, fs), config,
        associations(case.get('formula', False)) if case.get('associations') else None,
        AcousticBackends(probe, fail_irapt if case.get('fault') == 'irapt_raises' else None))
    frame = result.to_dataframe()
    assert list(frame.columns) == old['columns']
    assert result.time_axis.tolist() == old['time_axis']
    assert result.sampling_rate == fs
    actual = {str(c): pack(frame[c].to_numpy()) for c in frame}
    assert not support.compare(old['tracks'], actual), support.compare(old['tracks'], actual)
    np.testing.assert_array_equal(data, before)
    assert not any(name.startswith('phonetic_toolbox') for name in sys.modules)
    if len(data):
        assert probe.calls == ([1] if case.get('fault') == 'reaper_invalid_argument' else [0])
    if case.get('fault') == 'irapt_raises':
        assert any(e['actual'] == 'praat_fallback' for e in result.backend_events)
    elif len(data):
        assert any(e['actual'] == 'irapt1' for e in result.backend_events)
    if case.get('fault') == 'reaper_invalid_argument':
        assert any(e['actual'] == 'reaper_python' for e in result.backend_events)


def unpack_reference(value):
    if isinstance(value, dict) and 'values' in value:
        return np.array(value['values'], dtype=value['dtype'])
    if isinstance(value, dict): return {k: unpack_reference(v) for k, v in value.items()}
    return value


@pytest.mark.parametrize('recipe', [r for r in support.RECIPES if r['mode'] == 'acoustic'], ids=lambda r: r['id'])
def test_p03_independent_reference(recipe, tmp_path):
    old = support.load_json(ROOT/'tests/fixtures/golden'/f"{recipe['id']}.json.gz")['scientific']
    path = support.create_fixture(tmp_path, recipe)
    if old['status'] == 'error':
        with pytest.raises(ValueError): wavfile.read(path)
        return
    fs, data = wavfile.read(path)
    result = analyze_audio(AudioInput(data, fs), AcousticConfig(), backends=AcousticBackends(native_probe(tmp_path)))
    assert list(result.to_dataframe().columns) == old['dataframe_columns']
    expected = unpack_reference(old['result'])
    for key, value in expected.items():
        if key == 'sampling_rate':
            assert result.sampling_rate == old['audio']['sample_rate_hz']
            assert value == 16000  # M01-D01 known old metadata bug, not a numeric tolerance exception
            continue
        actual = getattr(result, key)
        if isinstance(value, dict):
            assert value.keys() == actual.keys()
            a, b = {k: pack(v) for k, v in value.items()}, {k: pack(v) for k, v in actual.items()}
        elif value is None:
            assert actual is None
            continue
        else:
            a, b = pack(value), pack(actual)
            if key == 'time_axis': np.testing.assert_array_equal(value, actual)
        assert not support.compare(a, b), (key, support.compare(a, b))


def test_consecutive_configs_are_isolated_and_cancellation_propagates(tmp_path):
    path = support.create_fixture(tmp_path, support.RECIPES[0])
    fs, data = wavfile.read(path)
    audio = AudioInput(data, fs)
    a = AcousticConfig(use_reaper=False, selected_parameter_keys=['pF0'])
    b = dataclasses.replace(a, frameshift_ms=10., selected_parameter_keys=['Intensity'])
    first = analyze_audio(audio, a).to_dataframe()
    second = analyze_audio(audio, b).to_dataframe()
    third = analyze_audio(audio, a).to_dataframe()
    assert list(first) == ['Time_s', 'pF0'] and list(second) == ['Time_s', 'Intensity']
    assert len(second) * 2 == len(first)
    assert first.equals(third)
    calls = []
    class Cancelled(Exception): pass
    def cancel():
        calls.append(True)
        if len(calls) == 4: raise Cancelled()
    with pytest.raises(Cancelled): analyze_audio(audio, a, cancellation=cancel)
    assert len(calls) == 4
