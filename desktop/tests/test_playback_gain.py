"""Playback PCM checks use an output stub, never a physical sound device."""
import base64
from types import SimpleNamespace
import numpy as np
import pytest
from phonetic_core.playback_gain import peak_gain,sample_peak,AUDITION_PEAK
from phonetic_core.vocal_tract.envelope import audition_samples
from ptb_worker.vocal_web import BrowserAudio


def test_gain_does_not_edit_source_and_bounds_silence_and_float_pcm():
    source=np.array([-.1,.05,.2]);original=source.copy();gain=peak_gain(sample_peak(source))
    np.testing.assert_array_equal(source,original)
    assert max(abs(source*gain))==pytest.approx(AUDITION_PEAK)
    assert peak_gain(0)==1 and peak_gain(.001)==32 and peak_gain(.01,False)==1
    assert peak_gain(2)*2<1


@pytest.mark.parametrize('enabled',[True,False])
@pytest.mark.parametrize('volume',[0.,.25,1.])
def test_web_native_stream_uses_fixed_gain_and_retains_volume_envelope(enabled,volume):
    engine=SimpleNamespace(params=np.zeros(2),default_f0=120.,sr=48000)
    player=BrowserAudio(engine);player.auto_gain=enabled;player.volume=volume
    source=np.full(1920,.01);source[960:]=.005;original=source.copy();envelope=np.linspace(0,1,1920)
    player.start(source,mode='preview',envelope=envelope)
    result=player.pull({});pcm=np.frombuffer(base64.b64decode(result['base64']),dtype='<f4')
    reference=audition_samples(source,volume,0,envelope)*peak_gain(sample_peak(audition_samples(source,1,0)),enabled)
    np.testing.assert_allclose(pcm,reference,rtol=1e-6,atol=1e-8)
    np.testing.assert_array_equal(source,original)
    assert np.max(np.abs(pcm))<1
