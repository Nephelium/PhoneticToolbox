from pathlib import Path
RESOURCES=Path(__file__).resolve().parents[2]/"resources/vocal_tract/native"
WEB=Path(__file__).resolve().parents[2]/"gui/resources/vocal_tract"
import io
import json
from pathlib import Path
import sys
import threading
from urllib import request,error
import numpy as np
import pytest
from scipy.io import wavfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phonetic_toolbox.services.vocal_tract.server import create_server

@pytest.fixture(scope='module')
def server(tmp_path_factory):
    s=create_server(playback_allowed=False,resource_dir=RESOURCES,web_dir=WEB,profile_dir=tmp_path_factory.mktemp('vocal-http'));thread=threading.Thread(target=s.serve_forever,daemon=True);thread.start()
    yield s
    s.shutdown();thread.join(3);s.app.live.stop();s.server_close();s.app.engine.close()

def call(s,path,obj=None,headers=None):
    h={'X-Session':s.app.token,'Content-Type':'application/json',**(headers or {})}
    data=json.dumps(obj).encode() if obj is not None else None
    req=request.Request(f'http://127.0.0.1:{s.server_port}'+path,data=data,headers=h)
    try:
        with request.urlopen(req,timeout=10) as r:return r.status,r.headers,r.read()
    except error.HTTPError as e:return e.code,e.headers,e.read()

def test_pose_revision_and_bad_input(server):
    p=server.app.engine.presets['a']
    code,_,raw=call(server,'/api/pose',{'params':p,'revision':1})
    assert code==200 and json.loads(raw)['geometry_area_max_error']<1e-8
    assert call(server,'/api/pose',{'params':p,'revision':1})[0]==409
    assert call(server,'/api/pose',{'params':[float('nan')]*19,'revision':2})[0]==400
    assert call(server,'/api/pose',{'params':p,'revision':2,'f0':900})[0]==400
    assert server.app.revision==1

def test_local_session_and_origin(server):
    assert call(server,'/api/live',{'active':True},{'X-Session':'invalid'})[0]==403
    assert call(server,'/api/meta',headers={'Origin':'https://example.com'})[0]==403
    assert call(server,'/api/meta',headers={'Host':'example.com'})[0]==403
    assert call(server,'/../engine.py')[0]==404
    assert call(server,'/vendor/VocalTractLabApi.dll')[0]==404

def test_render_returns_real_wav(server):
    code,headers,raw=call(server,'/api/render',{'params':server.app.engine.presets['u'],'duration':.3})
    assert code==200 and headers['Content-Type']=='audio/wav'
    sr,audio=wavfile.read(io.BytesIO(raw))
    assert sr==48000 and audio.shape==(14400,)
    assert np.isfinite(audio).all() and np.sqrt(np.mean(audio**2))>.001
    assert call(server,'/api/status')[0]==200

def test_audio_routes_share_one_playback_controller(server,monkeypatch):
    calls=[]
    def start(buffer=None,mode='live'):calls.append((mode,buffer))
    monkeypatch.setattr(server.app.live,'start',start)
    monkeypatch.setattr(server.app.live,'playback_allowed',True)
    assert call(server,'/api/live',{'active':True})[0]==200
    assert call(server,'/api/preview',{'duration':.3})[0]==200
    assert call(server,'/api/audio/test',{})[0]==200
    assert [mode for mode,_ in calls]==['live','preview','test']
    assert calls[0][1] is None
    assert np.sqrt(np.mean(calls[1][1]**2))>.001
    assert len(calls[2][1])==24000
    assert call(server,'/api/audio/settings',{'device':'not-an-output-device'})[0]==400
    assert call(server,'/api/audio/settings',{'volume':-1})[0]==400
    meta=json.loads(call(server,'/api/meta')[2])
    assert meta['api_version']==4 and 'output_devices' in meta

def test_classroom_mode_blocks_every_playback_route(server,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Playback controller must not be reached')
    monkeypatch.setattr(server.app.live,'start',forbidden)
    for path,body in [('/api/live',{'active':True}),('/api/preview',{}),('/api/audio/test',{}),('/api/audio/settings',{'volume':1})]:
        code,_,raw=call(server,path,body)
        assert code==403 and '静音' in json.loads(raw)['error']
    state=json.loads(call(server,'/api/status')[2])
    assert state['audio_locked'] and not state['active']
    assert call(server,'/api/live',{'active':False})[0]==200
    assert json.loads(call(server,'/api/meta')[2])['output_devices']==[]
    with pytest.raises(PermissionError):server.app.live.test_tone()

def test_motion_endpoints_prepare_without_playing_and_finish_at_last_pose(server,monkeypatch):
    frames=[{'params':server.app.engine.presets[n],'duration':.15,'lip_width':w,'f0':f} for n,w,f in [('a',.7,110),('u',1.3,170)]]
    code,_,raw=call(server,'/api/animation/prepare',{'frames':frames})
    assert code==200;prepared=json.loads(raw)
    assert not server.app.live.active
    assert call(server,'/api/animation/play',{'id':prepared['id']})[0]==403
    monkeypatch.setattr(server.app.live,'playback_allowed',True)
    calls=[]
    monkeypatch.setattr(server.app.live,'start',lambda audio,mode:calls.append((audio,mode)))
    assert call(server,'/api/animation/play',{'id':-1})[0]==400
    assert call(server,'/api/animation/play',{'id':prepared['id']})[0]==200
    assert calls[0][1]=='animation' and len(calls[0][0])==14400
    monkeypatch.setattr(server.app.live,'position',lambda:prepared['duration'])
    monkeypatch.setattr(server.app.live,'completed',True)
    result=json.loads(call(server,'/api/animation/frame',{'id':prepared['id']})[2])
    assert result['completed'] and not result['active']
    assert result['state']['lip_width']==1.3 and result['state']['f0']==170
    np.testing.assert_allclose(result['state']['params'],frames[-1]['params'])
    np.testing.assert_allclose(server.app.live.params,frames[-1]['params'])
    assert call(server,'/api/animation/stop',{})[0]==200

def test_pose_width_and_native_vowel_guard(server):
    p=list(server.app.engine.presets['o']);p[server.app.engine.names.index('VO')]=1.5
    revision=server.app.revision+1
    assert call(server,'/api/pose',{'params':p,'revision':revision,'lip_width':2})[0]==400
    assert server.app.revision<revision
    code,_,raw=call(server,'/api/pose',{'params':p,'revision':revision,'lip_width':1.2,'keep_vowel':True})
    assert code==200;state=json.loads(raw)
    assert state['lip_width']==1.2 and state['nasal_constraint']
    assert state['oral_min_area']>=.08
