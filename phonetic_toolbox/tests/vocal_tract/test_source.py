import json
import zlib
import numpy as np
import pytest
from scipy.signal import correlate
from phonetic_toolbox.models.vocal_source import SOURCE_PRESETS
from phonetic_toolbox.core.vocal_tract.source import validate_source,interpolate_source
from phonetic_toolbox.services.vocal_tract.animation import prepare_animation
from .test_http import server,call


def test_source_units_and_zero_vibration(server):
    engine=server.app.engine
    g=engine.glottis(210,source=SOURCE_PRESETS['whisper']);values=dict(zip(engine.gnames,g))
    assert values['pressure']==8000 and values['rel_amp']==0 and values['flutter']==0
    assert values['x_bottom']==pytest.approx(.005) and values['chink_area']==.04
    assert values['f0']==125 # do not let a hand-drawn F0 alter a non-vibrating source
    for source in [{'mode':'wrong'},{'pressure_pa':1601},{'opening_mm':float('nan')},{'mode':'whisper','vibration':1}]:
        with pytest.raises(ValueError):validate_source(source)


def test_native_unvoiced_noise_loses_periodicity_and_responds_to_pressure(server):
    engine=server.app.engine;tube=engine.prepare_tube(engine.presets['a'])
    def render(source):
        g=engine.glottis(source=source)
        with engine.lock:
            engine.reset_tube(tube,g)
            return np.concatenate([engine.block_tube(tube,g,960) for _ in range(35)])[9600:]
    def periodicity(y):
        a=correlate(y-y.mean(),y-y.mean(),method='fft')[len(y)-1:]
        return (a[137:800]/a[0]).max()
    voiced=render(SOURCE_PRESETS['voiced']);unvoiced=render(SOURCE_PRESETS['voiceless']);whisper=render(SOURCE_PRESETS['whisper'])
    assert periodicity(voiced)>.9 and periodicity(unvoiced)<.5 and periodicity(whisper)<.4
    assert np.sqrt(np.mean(whisper**2))>1e-5
    strong=render({**SOURCE_PRESETS['whisper'],'pressure_pa':1400})
    assert np.sqrt(np.mean(strong**2))>np.sqrt(np.mean(whisper**2))*1.3
    off=render({**SOURCE_PRESETS['whisper'],'pressure_pa':0})
    closed=render({**SOURCE_PRESETS['whisper'],'opening_mm':0,'posterior_gap_mm2':0})
    assert np.max(abs(off))<1e-9 and np.max(abs(closed))<1e-9


def test_source_transitions_are_continuous_and_survive_keyframe_storage(server):
    a=SOURCE_PRESETS['voiced'];b=SOURCE_PRESETS['whisper'];mid=interpolate_source(a,b,.5)
    assert mid['mode']=='transition' and mid['vibration']==.5
    f={'params':server.app.engine.presets['a'],'lip_width':1,'f0':125,'duration':.3}
    frames=[{**f,'source':a},{**f,'source':b}]
    assert call(server,'/api/keyframes',{'frames':frames})[0]==200
    saved=json.loads(call(server,'/api/keyframes')[2])['frames']
    assert saved[1]['source']==b
    animation=prepare_animation(server.app.engine,frames)
    states=[json.loads(zlib.decompress(p)) for p in animation['pictures']]
    assert states[0]['source']['vibration']==1 and states[-1]['source']['vibration']==0
    assert any(0<s['source']['vibration']<1 for s in states)
    assert np.isfinite(animation['audio']).all()


def test_pose_accepts_source_without_enabling_physical_playback(server):
    code,_,raw=call(server,'/api/pose',{'params':server.app.engine.presets['a'],'revision':server.app.revision+1,'source':SOURCE_PRESETS['whisper']})
    assert code==200 and json.loads(raw)['source']['vibration']==0
    assert server.app.live.source['mode']=='whisper' and not server.app.live.active
    assert call(server,'/api/live',{'active':True})[0]==403
