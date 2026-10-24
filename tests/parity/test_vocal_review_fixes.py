"""Native regressions for the five M10 review findings, without audio files/devices."""
import json
import os
import threading
import zlib
from pathlib import Path

import numpy as np
import pytest

from phonetic_core.vocal_tract.animation import prepare_animation
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.session import Runtime
from ptb_worker.vocal_web import BrowserAudio, MemoryProfile
from ptb_worker.browser_vocal import BrowserVocalRuntime

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=['JD2', 'W02'])
def engine(request):
    value = Engine(request.param, resource_dir=Path(os.environ.get('PTB_M10_TEST_RESOURCES', ROOT / 'resources/vocal_tract/native')))
    value.configure_perturbation({'jitter': 0., 'shimmer': 0., 'seed': 23})
    yield value
    value.close()


def pose(engine, **changes):
    return dict(params=list(engine.presets['a']), lip_width=1., f0=engine.default_f0,
                duration=.3, frame_id='a', **changes)


def feature(parameter, value, start=.1, end=.2):
    return dict(id=parameter, enabled=True, parameter=parameter, mode='fixed', value=value,
                selection=dict(kind='time', start=start, end=end))


@pytest.mark.parametrize('parameter,value', [('f0', 300.), ('pressure_pa', 1200.), ('lip_width', 1.4)])
def test_regional_parameters_do_not_change_raw_or_audition_pcm_before_start(engine, parameter, value):
    frame = pose(engine)
    dry = prepare_animation(engine, [frame], pictures_enabled=False)
    wet = prepare_animation(engine, [frame], features=[feature(parameter, value)], pictures_enabled=False)
    for key in ('audio', 'audition_audio'):
        np.testing.assert_array_equal(wet[key][:4800], dry[key][:4800])
    assert np.max(np.abs(wet['audio'][4800:9600] - dry['audio'][4800:9600])) > 1e-6


def test_overlapping_f0_features_switch_at_sample_rounded_half_open_boundaries(engine, monkeypatch):
    original = engine.block_tube
    calls = []
    offset = 0

    def record(tube, glottis, n=960):
        nonlocal offset
        calls.append((offset, n, float(glottis[0])))
        result = original(tube, glottis, n)
        offset += n
        return result

    monkeypatch.setattr(engine, 'block_tube', record)
    prepare_animation(engine, [pose(engine)], pictures_enabled=False, features=[
        feature('f0', 240., .100007, .200007),
        {**feature('f0', 280., .150007, .200007), 'id': 'second'},
    ])
    positive = [(start, size, f0) for start, size, f0 in calls if size]
    for start, size, f0 in positive:
        expected = engine.default_f0 if start < 4800 or start >= 9600 else 240. if start < 7200 else 280.
        assert f0 == expected, (start, size, f0, expected)
    for edge, expected in [(4800, 240.), (7200, 280.), (9600, engine.default_f0)]:
        assert (edge, 0, expected) in calls


@pytest.mark.parametrize('outcome', ['success', 'cancel', 'exception'])
@pytest.mark.parametrize('manual', [False, True])
def test_render_restores_manual_root_on_every_exit(engine, monkeypatch, outcome, manual):
    engine.set_manual_root(manual)
    cancel = threading.Event()
    original = engine.block_tube

    def interrupted(tube, glottis, n=960):
        if n and outcome == 'exception':
            raise RuntimeError('controlled native failure')
        result = original(tube, glottis, n)
        if n and outcome == 'cancel':
            cancel.set()
        return result

    monkeypatch.setattr(engine, 'block_tube', interrupted)
    frame = pose(engine, manual_root=not manual)
    if outcome == 'success':
        prepare_animation(engine, [frame], cancel=cancel)
    else:
        with pytest.raises(ValueError if outcome == 'cancel' else RuntimeError,
                           match='取消' if outcome == 'cancel' else 'controlled native failure'):
            prepare_animation(engine, [frame], cancel=cancel)
    assert engine.manual_root is manual


def test_live_start_synchronizes_runtime_root_mode_and_current_tract_compensation(engine):
    runtime = Runtime(engine, MemoryProfile(), BrowserAudio(engine))
    runtime.invoke('pose', dict(params=engine.presets['a'], revision=1, f0=240., manual_root=False))
    engine.set_manual_root(True)
    runtime.invoke('live', {'active': True})
    assert engine.manual_root is False
    first = runtime.live.pitch_gain
    runtime.invoke('pose', dict(params=engine.presets['i'], revision=2, f0=240., manual_root=False))
    expected = engine.pitch_level_gain(runtime.live.tube, 240., runtime.live.source)
    assert runtime.live.pitch_gain == expected
    assert expected != first
    runtime.live.stop()


def test_final_video_timeline_includes_global_regional_curve_and_unvoiced_states(engine):
    frame = pose(engine)
    result = prepare_animation(engine, [frame], pitch_curve=[[0., 100.], [1., 200.]],
                               global_controls={'f0': {'mode': 'fixed', 'value': 240.}},
                               features=[feature('f0', 280.)])
    points = result['timeline']
    assert points[0] == {'time': 0., 'f0': 240.}
    assert next(p for p in points if p['time'] == .1)['f0'] == 280.
    assert next(p for p in points if p['time'] == .2)['f0'] == 240.
    for time, picture in zip(result['times'], result['pictures']):
        state = json.loads(zlib.decompress(picture))
        assert state['f0'] == (280. if .1 <= time < .2 else 240.)
    quiet = prepare_animation(engine, [{**frame, 'source': {'mode': 'voiceless'}}])
    assert all(p['f0'] is None for p in quiet['timeline'])
    silent = prepare_animation(engine, [{**frame, 'silent': True}])
    assert all(p['f0'] is None for p in silent['timeline'])


def test_session_passes_the_final_timeline_to_every_animation_cache_hit(engine):
    runtime = Runtime(engine, MemoryProfile(), BrowserAudio(engine))
    body = dict(frames=[pose(engine)], global_controls={'f0': {'mode': 'fixed', 'value': 240.}})
    first = runtime.invoke('animation/prepare', body)
    second = runtime.invoke('animation/prepare', body)
    assert not first['cached'] and second['cached']
    assert first['timeline'] == second['timeline'] == runtime.animation['timeline']


@pytest.mark.parametrize('operation', ['animation/stop', 'deactivate', 'live'])
def test_stop_adopts_the_current_complete_animation_pose(engine, operation):
    runtime = Runtime(engine, MemoryProfile(), BrowserAudio(engine))
    runtime.invoke('pose', dict(params=engine.presets['a'], revision=1, f0=120.))
    frames = [dict(pose(engine), params=engine.presets['i'], f0=200., manual_root=True,
                   lip_width=1.2, larynx_height=.1, tongue_shape=[.1, 0., 0., 0.],
                   lip_shape=[0., 0., 0., .1]),
              dict(pose(engine), params=engine.presets['u'], f0=240., manual_root=True, frame_id='u')]
    clip = runtime.invoke('animation/prepare', dict(frames=frames))
    runtime.invoke('animation/play', {'id': clip['id']})
    runtime.live.pull({})
    runtime.live.acknowledge({'position': .1})
    shown = runtime.invoke('animation/frame', {'id': clip['id']})['state']
    stopped = runtime.invoke(operation, {'active': False})
    assert stopped['pose'] == shown
    assert runtime.manual_root is engine.manual_root is shown['manual_root']
    for key in ('params', 'lip_width', 'f0', 'source', 'larynx_height', 'tongue_shape', 'lip_shape'):
        np.testing.assert_equal(getattr(runtime.live, key), shown[key])
    runtime.invoke('live', {'active': True})
    assert runtime.live.pitch_gain == engine.pitch_level_gain(runtime.live.tube, shown['f0'], shown['source'])
    runtime.live.stop()


@pytest.mark.parametrize('manual', [False, True])
@pytest.mark.parametrize('outcome', ['success', 'cache', 'preprocess_error', 'render_error', 'cancel'])
def test_keep_vowel_preprocessing_restores_the_original_mode_on_every_exit(engine, monkeypatch, manual, outcome):
    runtime = Runtime(engine, MemoryProfile(), BrowserAudio(engine))
    runtime.invoke('pose', dict(params=engine.presets['a'], revision=1, manual_root=manual))
    body = dict(frames=[pose(engine, manual_root=not manual)], keep_vowel=True)
    if outcome == 'cache':
        runtime.invoke('animation/prepare', body)
        engine.set_manual_root(manual)
    original = engine.block_tube
    def fail_preprocess(*args):
        raise RuntimeError('controlled preprocessing failure')
    def block(tube, glottis, n=960):
        if n and outcome == 'render_error':
            raise RuntimeError('controlled rendering failure')
        value = original(tube, glottis, n)
        if n and outcome == 'cancel':
            runtime.cancel.set()
        return value
    monkeypatch.setattr(engine, 'block_tube', block)
    if outcome == 'preprocess_error':
        monkeypatch.setattr(engine, 'constrain_nasal_opening', fail_preprocess)
    if outcome in ('success', 'cache'):
        assert runtime.invoke('animation/prepare', body)['cached'] is (outcome == 'cache')
    else:
        with pytest.raises(ValueError if outcome == 'cancel' else RuntimeError):
            runtime.invoke('animation/prepare', body)
    assert engine.manual_root is runtime.manual_root is manual
    assert 'pose' not in runtime.invoke('animation/stop')


@pytest.mark.parametrize('completed', [False, True])
def test_browser_stop_and_completion_acknowledge_final_clock_and_checkpoint_the_adopted_pose(engine, completed):
    host=BrowserVocalRuntime.__new__(BrowserVocalRuntime)
    host.runtime=Runtime(engine,MemoryProfile(),BrowserAudio(engine))
    host.invoke('pose',dict(params=engine.presets['a'],revision=1,f0=120.))
    clip=host.invoke('animation/prepare',dict(frames=[dict(pose(engine),params=engine.presets['i'],f0=240.,manual_root=True)]))
    host.invoke('animation/play',{'id':clip['id']});host.invoke('audio/pull',{})
    result=json.loads(host.dispatch('animation/frame' if completed else 'animation/stop',json.dumps(dict(id=clip['id'],position=.3 if completed else .1,completed=completed))))
    state=result['value']['state' if completed else 'pose']
    assert result['checkpoint']['pose']['f0']==state['f0']==host.runtime.live.f0==240.
    assert host.runtime.live.position()==(.3 if completed else .1)
    assert result['checkpoint']['pose']['manual_root'] is engine.manual_root is True
    np.testing.assert_array_equal(result['checkpoint']['pose']['params'],state['params'])
