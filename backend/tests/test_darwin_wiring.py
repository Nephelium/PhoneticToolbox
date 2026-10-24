"""Platform-independent contracts; real Darwin process tests remain separate."""
import io
import json
from types import SimpleNamespace

import pytest


def test_mac_persistent_egg_keeps_owner_and_current_optional_flags():
    from ptb_worker.native.darwin_egg import InteractivePreview
    from ptb_worker.spectrogram_preview import PreviewError
    session=InteractivePreview()
    calls=[]
    session.owner='owner';session.token='token';session.process=object()
    session._rpc=lambda header,**kw:calls.append((header,kw)) or {'ok':True}
    try:
        with pytest.raises(PreviewError,match='egg_preview_expired'):
            session.update('other','token',{})
        assert not calls
        assert session.update('owner','token',{'spec_vmin':-60},auto_db=True,need_audio=True)=={'ok':True}
        assert calls==[({'config':{'spec_vmin':-60},'auto_db':True,'need_audio':True},{'timeout':30})]
    finally:
        session.process=None;session.close()


def test_persistent_egg_protocol_keeps_requested_audio_during_auto_range(monkeypatch):
    from ptb_worker import egg_interactive_child as child
    updates=[]
    class Session:
        def __init__(self,raw,**kwargs):self.sha='source';self.audio_key='already-produced'
        def update(self,config):
            updates.append((dict(config),self.audio_key))
            audio='audio' if self.audio_key is None else None
            self.audio_key='already-produced'
            return {'preview':{'suggested_db_range':[-55,-5]},'audio_base64':audio,'config':config}
    raw=b'{"size":1}\nx'+json.dumps({'config':{'spec_vmin':-60},'auto_db':True,'need_audio':True}).encode()+b'\n'
    output=io.BytesIO();stdout=io.TextIOWrapper(output,encoding='utf-8')
    monkeypatch.setattr(child,'Session',Session)
    monkeypatch.setattr(child,'sys',SimpleNamespace(stdin=SimpleNamespace(buffer=io.BytesIO(raw)),stdout=stdout))
    child.run();stdout.flush();data=output.getvalue()
    first_size,_,rest=data.partition(b'\n');rest=rest[int(first_size):]
    size,_,payload=rest.partition(b'\n');value=json.loads(payload[:int(size)])
    assert updates==[({'spec_vmin':-60},None),({'spec_vmin':-55,'spec_vmax':-5},'already-produced')]
    assert value['audio_base64']=='audio' and value['config']['spec_vmax']==-5


def test_spectrogram_mac_delegate_keeps_windows_cache_state(monkeypatch):
    from ptb_worker import spectrogram_session as host
    from ptb_worker.native import darwin_spectrogram
    calls=[]
    class Delegate:
        def __init__(self,idle):calls.append(('idle',idle))
        def render(self,payload,**options):calls.append(('render',payload,options));return {'pixels':'original'}
        def close(self):calls.append(('closed',))
    monkeypatch.setattr(darwin_spectrogram,'SpectrogramSession',Delegate)
    import sys
    monkeypatch.setattr(sys,'platform','darwin')
    session=host.SpectrogramSession(idle_seconds=7)
    try:assert session.render(b'audio',channel=0,start=0,end=1)=={'pixels':'original'}
    finally:session.close()
    assert session.cached is None
    assert calls[0]==('idle',7) and calls[-1]==('closed',)
