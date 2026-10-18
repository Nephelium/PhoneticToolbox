"""Real VTL worker/HTTP checks; account double is explicitly not PostgreSQL evidence."""
import base64
from pathlib import Path
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from ptb_api.main import create_app
from ptb_api.auth import AuthSettings
from ptb_worker.vocal_sessions import VocalSessions,VocalError
from account_double import MemoryAccountStore

ROOT=Path(__file__).resolve().parents[2]
RESOURCES=ROOT/'resources/vocal_tract/native'
ORIGIN='https://vocal.test'


@pytest.fixture
def service():
    value=VocalSessions(RESOURCES)
    yield value
    value.close()


def test_web_sections_real_geometry_and_server_rejection(service):
    ident=('alice','lease','project');sid=service.open(ident)['id'];call=lambda op,body={}:service.invoke(sid,ident,op,body)
    meta=call('meta');assert meta['section_count']==33 and meta['section_options']==[33,65]
    states=[]
    for revision,count in enumerate([33,65,33],1):
        state=call('pose',dict(params=meta['presets']['a'],revision=revision,section_count=count))
        assert len(state['centerline'])==len(state['airway_sections'])==count
        assert np.isfinite(state['tube_areas']).all();states.append(state)
    assert states[0]['tube_areas']==states[2]['tube_areas']
    assert states[0]['contours']==states[1]['contours']
    for count in [129,257,True,33.0,0]:
        with pytest.raises(VocalError):call('pose',dict(params=meta['presets']['a'],revision=10,section_count=count))
    assert call('status')['revision']==3
    with pytest.raises(VocalError):service.get(sid,('bob','lease','project'))


def test_browser_stream_matches_desktop_test_tone_and_monitor(service):
    ident=('alice','lease','project');sid=service.open(ident)['id'];call=lambda op,body={}:service.invoke(sid,ident,op,body)
    started=call('audio/test');parts=[];position=0
    stale=call('audio/pull',dict(position=99,completed=True,playback_generation=started['playback_generation']-1))
    assert stale['samples']==0 and stale['eof']
    while True:
        packet=call('audio/pull',{'position':position});audio=np.frombuffer(base64.b64decode(packet['base64']),'<f4');parts.append(audio);position+=len(audio)/48000
        if packet['eof']:break
    actual=np.concatenate(parts);t=np.arange(24000)/48000
    expected=(np.tanh(.035*np.sin(2*np.pi*440*t)*np.minimum(1,t/.02)*np.minimum(1,(.5-t)/.02)*4*.8)*.9).astype('<f4')
    np.testing.assert_array_equal(actual,expected)
    state=call('heartbeat',dict(position=.5,completed=True));assert not state['active']
    monitor=call('audio/monitor',dict(seconds=1,position=.5,completed=True))
    assert monitor['end']==.5 and monitor['sample_rate']==48000
    assert max(abs(v) for row in monitor['waveform'] for v in row)>0


def test_profiles_document_frames_audio_and_cleanup(service):
    ident=('alice','lease','project');sid=service.open(ident)['id'];call=lambda op,body={}:service.invoke(sid,ident,op,body)
    meta=call('meta');frame=dict(params=meta['presets']['a'],name='测试元音',duration=.05,lip_width=1,f0=150,tongue_shape=[0,0,0,0])
    presets=call('presets/save',{'presets':[{**frame,'id':'custom-a'}]});assert presets[0]['name']=='测试元音'
    assert call('favorites/save',{'ids':['custom-a','i']})==['custom-a','i']
    call('keyframes',{'frames':[frame],'pitch_curve':[[0,100],[1,200]]})
    document=call('document/create',call('keyframes/load'));assert document['section_count']==33
    assert call('document/validate',document)==document
    prepared=call('animation/prepare',call('keyframes/load'));assert prepared['duration']==.05
    audio=call('animation/audio',{'id':prepared['id']});assert audio['samples']==2400
    assert np.std(np.frombuffer(base64.b64decode(audio['base64']),'<f4'))>0
    assert call('animation/picture',{'id':prepared['id'],'index':0})['state']['section_count']==33
    child=service.get(sid,ident);scratch=Path(child.scratch.name);pid=child.process.pid
    service.release(sid,ident);assert child.process.poll() is not None and not scratch.exists()


def test_web_bounded_audition_pose_lock_and_five_second_limit(service):
    ident=('alice','bounded','project');sid=service.open(ident)['id'];call=lambda op,body={}:service.invoke(sid,ident,op,body)
    meta=call('meta');assert meta['continuous_audio'] is False and meta['max_animation_duration']==5
    with pytest.raises(VocalError,match='持续发声'):call('live',{'active':True})
    for body in ({'duration':2},{'duration':1},{'duration':.5,'sequence':'a-i-u'}):
        with pytest.raises(VocalError,match='0.5 秒测试发声'):call('preview',body)
    started=call('preview',{'duration':.5});assert started['active']
    with pytest.raises(VocalError,match='等待测试发声'):call('pose',dict(params=meta['presets']['i'],revision=1))
    packets=[];position=0
    while True:
        packet=call('audio/pull',dict(position=position));packets.append(packet['samples']);position+=packet['samples']/48000
        if packet['eof']:break
    assert packets==[23040,960] and position==.5
    assert not call('heartbeat',dict(position=.5,completed=True))['active']
    assert call('pose',dict(params=meta['presets']['i'],revision=1))['revision']==1
    frames=[dict(params=meta['presets'][name],duration=1.25) for name in ('a','i','u','a')]
    assert call('animation/prepare',{'frames':frames})['duration']==5
    frames[-1]['duration']=1.26
    with pytest.raises(VocalError,match='最多 5 秒'):call('animation/prepare',{'frames':frames})
    video=call('video/begin',dict(width=1280,height=720,fps=30,duration=60))
    assert call('video/cancel',{'id':video['id']})['cancelled']
    with pytest.raises(VocalError,match='最多 60 秒'):call('video/begin',dict(width=1280,height=720,fps=30,duration=60.01))


def test_http_account_project_csrf_origin_and_disabled_service():
    store=MemoryAccountStore();store.create_user('alice','a-correct-long-password');store.create_user('bob','b-correct-long-password')
    settings=AuthSettings(origin=ORIGIN,signing_key='vocal-test-'*8)
    app=create_app(account_store=store,auth_settings=settings,vocal_resources=RESOURCES)
    def login(client,name,password):
        token=client.get('/api/v1/auth/challenge').json()['csrf_token']
        r=client.post('/api/v1/auth/login',json=dict(username=name,password=password),headers={'Origin':ORIGIN,'X-CSRF-Token':token});assert r.status_code==200
        return {'Origin':ORIGIN,'X-CSRF-Token':r.json()['csrf_token'],'X-PTB-Account':r.json()['user']['id']}
    with TestClient(app,base_url=ORIGIN) as alice,TestClient(app,base_url=ORIGIN) as bob:
        ah=login(alice,'alice','a-correct-long-password');bh=login(bob,'bob','b-correct-long-password')
        project=alice.post('/api/v1/projects',json={'name':'vocal'},headers=ah).json()['id'];root='/api/v1/vocal/sessions?project_id='+project
        assert alice.post(root,json={}).status_code==403
        assert bob.post(root,json={},headers=bh).status_code==404
        opened=alice.post(root,json={},headers=ah);assert opened.status_code==200 and opened.headers['cache-control']=='no-store'
        sid=opened.json()['id'];path='/api/v1/vocal/sessions/'+sid+'?project_id='+project
        assert alice.post(path,json={'op':'meta'},headers={**ah,'Origin':'https://elsewhere.test'}).status_code==403
        assert alice.post(path,json={'op':'meta','body':{'path':'/etc/passwd'}},headers=bh).status_code==409
        assert alice.post(path,json={'op':'unapproved-operation'},headers=ah).status_code==422
        assert alice.post(path,json={'op':'meta'},headers=ah).json()['section_count']==33
        assert alice.delete(path,headers=ah).status_code==200
        assert alice.post(path,json={'op':'meta'},headers=ah).status_code==404


def test_idle_expiry_capacity_and_revocation():
    allowed=[True];v=VocalSessions(RESOURCES,max_sessions=1,idle_seconds=2,valid=lambda identity:allowed[0])
    try:
        identity=('a','s','p');sid=v.open(identity)['id'];child=v.get(sid,identity)
        with pytest.raises(VocalError,match='capacity'):v.open(('b','s','p'))
        child.touched=time.monotonic()-3;v.sweep();assert child.process.poll() is not None
        sid=v.open(identity)['id'];child=v.get(sid,identity);allowed[0]=False;v.sweep();assert child.process.poll() is not None
    finally:v.close()


def test_stop_cancels_a_render_waiting_for_compute_slot(service):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    identity=('a','s','p');sid=service.open(identity)['id'];meta=service.invoke(sid,identity,'meta',{})
    original=service.compute;waiting=threading.Event();original.acquire()
    class Gate:
        def acquire(self,timeout):waiting.set();return original.acquire(timeout=timeout)
        def release(self):original.release()
    service.compute=Gate()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(service.invoke,sid,identity,'animation/prepare',{'frames':[dict(params=meta['presets']['a'],duration=.05)]})
        assert waiting.wait(2)
        assert service.invoke(sid,identity,'heartbeat',{})['revision']==0
        service.invoke(sid,identity,'animation/stop',{})
        original.release()
        with pytest.raises(VocalError,match='vocal_cancelled'):future.result(timeout=5)
