"""M10 playback keeps its PCM clock while adapting shared WASAPI endpoints."""
from types import SimpleNamespace

import numpy as np
import pytest

from ptb_desktop.vocal_tract import audio_output as audio


@pytest.mark.parametrize('host,rate,channels', [
    ('Windows WASAPI', 44100, 2), ('Windows WASAPI', 48000, 1),
    ('ALSA', 48000, 2),
])
@pytest.mark.parametrize('mode', ['test', 'preview', 'animation'])
def test_buffer_playback_preserves_samples_and_clock(monkeypatch, host, rate, channels, mode):
    engine=SimpleNamespace(sr=48000,params=np.zeros(19),glottis=lambda *a,**k:None)
    device={'id':host+'|test','index':7,'name':'test','channels':channels}
    monkeypatch.setattr(audio,'output_devices',lambda:[device])
    monkeypatch.setattr(audio.sd,'WasapiSettings',lambda **kw:SimpleNamespace(**kw))
    writes=[]
    class Stream:
        latency=.04
        def __init__(self,**kw):
            assert kw['device']==7 and kw['samplerate']==48000
            extra=kw['extra_settings']
            if host=='Windows WASAPI':
                assert extra.auto_convert and not extra.exclusive
            else:assert extra is None
            if rate!=kw['samplerate'] and not (extra and extra.auto_convert):
                raise RuntimeError('Invalid sample rate')
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def write(self,block):writes.append(block.copy());return False
    monkeypatch.setattr(audio.sd,'OutputStream',Stream)
    player=audio.LiveAudio(engine,playback_allowed=True)
    samples=np.linspace(-.05,.05,1440)
    player.start(samples,mode=mode)
    player.thread.join(2)
    assert not player.thread.is_alive() and player.completed and player.error is None
    assert player.position()==pytest.approx(.03)
    actual=np.concatenate(writes)
    expected=audio.audition_samples(np.pad(samples,(0,480)),.8,0).astype(np.float32)
    assert actual.shape==(1920,channels)
    for ch in range(channels):np.testing.assert_array_equal(actual[:,ch],expected)
    assert engine.sr==48000


def test_output_start_failure_is_visible_and_thread_stops(monkeypatch):
    device={'id':'Windows WASAPI|test','index':7,'name':'test','channels':2}
    monkeypatch.setattr(audio,'output_devices',lambda:[device])
    monkeypatch.setattr(audio.sd,'WasapiSettings',lambda **kw:None)
    def fail(**kw):raise RuntimeError('device disconnected during opening')
    monkeypatch.setattr(audio.sd,'OutputStream',fail)
    engine=SimpleNamespace(sr=48000,params=np.zeros(19),glottis=lambda *a,**k:None)
    player=audio.LiveAudio(engine,playback_allowed=True)
    player.start(np.zeros(960));player.thread.join(2)
    assert not player.active and not player.completed
    assert player.error=='device disconnected during opening'
    player.stop()
