from types import SimpleNamespace
import pytest
from ptb_worker import egg_interactive as module
from ptb_worker.spectrogram_preview import PreviewError

def test_linux_session_owns_source_token_and_releases(monkeypatch):
    monkeypatch.setattr(module,'sys',SimpleNamespace(platform='linux'))
    service=module.InteractivePreview(idle_seconds=600)
    calls=[]
    def rpc(raw,config=None):
        calls.append((raw,config));return {'sha256':'a'} if config is None else {'preview':config}
    monkeypatch.setattr(service,'_linux_rpc',rpc)
    try:
        token=service.open('owner-a',b'stereo')['session_id']
        with pytest.raises(PreviewError,match='preview_busy'):service.open('owner-b',b'other')
        with pytest.raises(PreviewError,match='egg_preview_expired'):service.update('owner-b',token,{})
        with pytest.raises(PreviewError,match='egg_preview_expired'):service.update('owner-a','wrong',{})
        assert service.update('owner-a',token,{'flip_channels':True})=={'preview':{'flip_channels':True}}
        service.release('owner-b',token);assert service.linux_raw==b'stereo'
        service.release('owner-a',token);assert service.linux_raw is None
        with pytest.raises(PreviewError,match='egg_preview_expired'):service.update('owner-a',token,{})
        assert calls==[(b'stereo',None),(b'stereo',{'flip_channels':True})]
    finally:service.close()

def test_linux_failed_update_keeps_owner_source_for_retry(monkeypatch):
    monkeypatch.setattr(module,'sys',SimpleNamespace(platform='linux'))
    service=module.InteractivePreview(idle_seconds=600)
    monkeypatch.setattr(service,'_linux_rpc',lambda raw,config=None:{'sha256':'a'})
    try:
        token=service.open('a',b'stereo')['session_id']
        def fail(*args):raise PreviewError('preview_timeout',503)
        monkeypatch.setattr(service,'_linux_rpc',fail)
        with pytest.raises(PreviewError,match='preview_timeout'):service.update('a',token,{})
        assert service.owner=='a' and service.linux_raw==b'stereo'
    finally:service.close()


def test_audio_ready_is_scoped_to_source_and_channel(monkeypatch,tmp_path):
    import json
    from ptb_worker.native import linux_runtime,posix
    monkeypatch.setattr(module,'sys',SimpleNamespace(platform='linux'))
    monkeypatch.setattr(linux_runtime,'load_profile',lambda:(tmp_path/'profile',{'cache':str(tmp_path)}))
    monkeypatch.setattr(linux_runtime,'command',lambda *args:['trusted-child'])
    headers=[]
    def run(argv,payload,*args,**kwargs):
        header,raw=payload.split(b'\n',1);h=json.loads(header);headers.append(h)
        return json.dumps({'sha256':'a'} if h['config'] is None else {'audio_base64':None if h['audio_ready'] else 'audio'}).encode()
    monkeypatch.setattr(posix,'run_bounded',run)
    service=module.InteractivePreview(idle_seconds=600)
    try:
        token=service.open('a',b'stereo')['session_id']
        service.update('a',token,{'flip_channels':False},auto_db=True)
        service.update('a',token,{'flip_channels':False})
        service.update('a',token,{'flip_channels':True})
        assert [h['audio_ready'] for h in headers[1:]]==[False,True,False]
        assert headers[1]['auto_db'] is True
        service.close();token=service.open('a',b'another-source')['session_id']
        service.update('a',token,{'flip_channels':True})
        assert headers[-1]['audio_ready'] is False
    finally:service.close()
