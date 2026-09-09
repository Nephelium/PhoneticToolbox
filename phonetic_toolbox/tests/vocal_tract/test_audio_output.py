from pathlib import Path
RESOURCES=Path(__file__).resolve().parents[2]/"resources/vocal_tract/native"
WEB=Path(__file__).resolve().parents[2]/"gui/resources/vocal_tract"
"""Routing regressions without emitting sound during the test suite."""
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phonetic_toolbox.services.vocal_tract import audio_output as output

@pytest.fixture
def playback(monkeypatch,tmp_path):
    hosts=[{'name':'Windows WASAPI'}]
    devices=[{'name':'Bluetooth headset','hostapi':0,'max_output_channels':2},
             {'name':'Speakers (Realtek Audio)','hostapi':0,'max_output_channels':2}]
    monkeypatch.setattr(output.sd,'query_hostapis',lambda:hosts)
    monkeypatch.setattr(output.sd,'query_devices',lambda:devices)
    engine=SimpleNamespace(params=np.zeros(19),sr=48000,lock=threading.Lock(),
        glottis=lambda f:np.zeros(1),reset_tube=lambda p,g:None,prepare_tube=lambda p,w:{'areas':np.ones(50)},
        block_tube=lambda p,g,n:.03*np.sin(np.arange(n)*2*np.pi*125/48000))
    live=output.LiveAudio(engine,playback_allowed=True,config_path=tmp_path/'output/audio-settings.json');writes=[];routes=[]
    class Stream:
        latency=.04
        def __init__(self,**kwargs):routes.append(kwargs)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def write(self,block):
            writes.append(block.copy())
            if live.mode=='live' and len(writes)>=4:live.active=False
            return False
    monkeypatch.setattr(output.sd,'OutputStream',Stream)
    yield live,devices,routes,writes
    live.stop()

def test_all_routes_resolve_selected_speakers_after_device_reordering(playback):
    live,devices,routes,writes=playback
    assert 'Realtek' in live.device_key
    devices.reverse()  # Device indices can change when Bluetooth disconnects.
    for mode in ['live','preview','test']:
        writes.clear()
        if mode=='test':live.test_tone()
        else:live.start(None if mode=='live' else np.ones(1920)*.02,mode=mode)
        live.thread.join(2)
        assert not live.active and live.error is None
        assert routes[-1]['device']==0 and routes[-1]['channels']==2
        assert writes and max(float(np.max(np.abs(b))) for b in writes)>.01
        for b in writes:
            assert b.dtype==np.float32 and b.shape==(960,2)
            np.testing.assert_array_equal(b[:,0],b[:,1])
            assert np.isfinite(b).all() and np.max(np.abs(b))<=.9

def test_invalid_settings_are_atomic_and_valid_settings_persist(playback):
    live,devices,_,_=playback;original=live.device_key
    with pytest.raises(ValueError):live.configure(device='missing')
    other=output.output_devices()[0]['id']
    with pytest.raises(ValueError):live.configure(device=other,volume=2)
    assert live.device_key==original and live.volume==.8
    live.configure(volume=.55)
    saved=json.loads(live.config.read_text(encoding='utf-8'))
    assert saved=={'device':original,'volume':.55,'gain_db':0.}
    live.configure(gain_db=12)
    with pytest.raises(ValueError):live.configure(device=other,gain_db=25)
    assert live.device_key==original and live.gain_db==12

def test_missing_selected_device_does_not_silently_fall_back(playback):
    live,devices,_,_=playback;devices.pop()
    with pytest.raises(ValueError,match='断开'):live.start()
    assert not live.active
