import json
import zlib
import numpy as np
import pytest
from phonetic_toolbox.core.vocal_tract.monitor import OutputHistory, analyze_output
from phonetic_toolbox.core.vocal_tract.trajectory import validate_pitch_curve, sample_trajectory
from phonetic_toolbox.services.vocal_tract.animation import prepare_animation
from .test_http import server, call
from .test_audio_output import playback


def test_ring_wrap_retains_order_and_reset_generation():
    ring=OutputHistory(10,seconds=2)
    ring.append(np.arange(13));ring.append(np.arange(13,38))
    samples,end,g=ring.snapshot(2)
    np.testing.assert_array_equal(samples,np.arange(18,38));assert end==3.8
    ring.reset();samples,end,new=ring.snapshot(2)
    assert len(samples)==0 and end==0 and new==g+1


def test_spectrum_frequency_amplitude_time_and_peak_preservation():
    sr=48000;t=np.arange(sr)/sr;y=.5*np.sin(2*np.pi*1000*t)
    result=analyze_output(y,sr,seconds=1,window_ms=40,hop_ms=5,end=1)
    spectrum=np.asarray(result['spectrogram']);peak=np.argmax(spectrum[len(spectrum)//2])
    assert abs(peak*result['frequency_step']-1000)<result['frequency_step']
    peak_db=spectrum[len(spectrum)//2,peak]/255*90-90
    assert -6.8<peak_db<-5.5
    assert result['frame_offset']==.02 and result['hop_seconds']==.005
    impulse=np.zeros(sr);impulse[137]=.92
    wave=analyze_output(impulse,sr,seconds=1)['waveform']
    assert max(p[1] for p in wave)==pytest.approx(.92)


def test_monitor_empty_silent_bounds_and_decimation():
    empty=analyze_output([],48000);assert not empty['waveform'] and not empty['spectrogram']
    quiet=analyze_output(np.zeros(48000),48000,seconds=1)
    assert not np.asarray(quiet['spectrogram']).any()
    t=np.arange(48000)/48000
    alias=analyze_output(.5*np.sin(2*np.pi*13000*t),48000,seconds=1,window_ms=40)
    # 13 kHz would alias to 3 kHz without the resampling low-pass filter.
    assert np.max(alias['spectrogram'])/255*90-90 < -55
    for bad in [0,13,float('nan')]:
        with pytest.raises(ValueError):analyze_output([],48000,seconds=bad)
    for window in [4,81]:
        with pytest.raises(ValueError):analyze_output([],48000,window_ms=window)
    bounded=analyze_output(np.zeros(12*48000),48000,seconds=12,window_ms=80,hop_ms=5)
    assert len(bounded['spectrogram'])<=320 and len(bounded['spectrogram'][0])<=256


def test_monitor_records_exact_samples_sent_to_mock_output(playback):
    live,devices,routes,writes=playback
    live.start(np.linspace(-.1,.1,2000));live.thread.join(2)
    actual,_,_=live.output_history.snapshot(1)
    np.testing.assert_allclose(actual,np.concatenate(writes)[:,0],atol=1e-7)


def test_pitch_curve_validation_and_duration_scaling():
    frames=[{'params':[0.], 'lip_width':.8,'f0':125.,'duration':.6},
            {'params':[1.], 'lip_width':1.2,'f0':140.,'duration':.6}]
    curve=validate_pitch_curve([[0,80],[.5,220],[1,100]])
    assert sample_trajectory(frames,.6,curve)['f0']==220
    assert sample_trajectory(frames,1.2,curve)['f0']==100
    doubled=[{**f,'duration':f['duration']*2} for f in frames]
    assert sample_trajectory(doubled,1.2,curve)['f0']==220
    assert sample_trajectory(frames,.6,[])['f0']==140
    for bad in [[[.1,100],[1,150]],[[0,100],[0,150],[1,200]],[[0,59],[1,350]],[[0,float('nan')],[1,100]],'bad']:
        with pytest.raises(ValueError):validate_pitch_curve(bad)


def test_curve_is_saved_and_invalid_curve_does_not_replace_it(server):
    frame={'params':server.app.engine.presets['a'],'lip_width':1,'f0':125,'duration':.3}
    curve=[[0,100],[.5,240],[1,120]]
    assert call(server,'/api/keyframes',{'frames':[frame,frame],'pitch_curve':curve})[0]==200
    assert json.loads(call(server,'/api/keyframes')[2])['pitch_curve']==curve
    assert call(server,'/api/keyframes',{'frames':[],'pitch_curve':[[0,0],[1,200]]})[0]==400
    assert json.loads(call(server,'/api/keyframes')[2])['pitch_curve']==curve
    assert call(server,'/api/audio/monitor',{'seconds':1})[0]==200
    assert call(server,'/api/audio/monitor',{'seconds':99})[0]==400
    assert call(server,'/api/audio/monitor',{}, {'X-Session':'wrong'})[0]==403


def test_native_audio_and_cached_display_share_authored_pitch(server,monkeypatch):
    engine=server.app.engine;seen=[];original=engine.glottis
    def glottis(f0=125,pressure=8000):
        seen.append(f0);return original(f0,pressure)
    monkeypatch.setattr(engine,'glottis',glottis)
    f={'params':engine.presets['a'],'lip_width':1,'f0':125,'duration':.3}
    curve=[[0,100],[.5,210],[1,150]]
    animation=prepare_animation(engine,[f,f],pictures_enabled=True,pitch_curve=curve)
    for time,packed in zip(animation['times'],animation['pictures']):
        state=json.loads(zlib.decompress(packed))
        assert state['f0']==pytest.approx(np.interp(time/.6,[0,.5,1],[100,210,150]))
    assert any(abs(p-210)<.001 for p in seen) and seen[-1]==pytest.approx(150)
    assert np.sqrt(np.mean(animation['audio']**2))>.001
