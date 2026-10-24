"""Platform selection contracts; physical Mac capture and Dock remain device tests."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ptb_desktop.m05_bridge import M05Bridge
from ptb_desktop.macos_updates import MacUpdateService
from ptb_desktop.updates import UpdateError
from ptb_desktop.recording import service as recording


@pytest.mark.parametrize('platform',['win32','darwin'])
def test_host_selects_platform_update_bridge_and_preserves_service_options(tmp_path,platform):
    source=Path(__file__).resolve().parents[1]/'src/ptb_desktop/host.py'
    tree=ast.parse(source.read_text(encoding='utf8'))
    choices=[node for node in ast.walk(tree) if isinstance(node,ast.If)
             and any(isinstance(child,ast.ImportFrom) and child.module=='macos_updates'
                     for child in node.body)]
    assert len(choices)==1
    scope={'sys':SimpleNamespace(platform=platform),'__package__':'ptb_desktop'}
    exec(compile(ast.Module(body=choices,type_ignores=[]),str(source),'exec'),scope)
    bridge=scope['UpdatesBridge'](data_root=tmp_path/'cache',state_root=tmp_path/'state')
    try:
        assert isinstance(bridge.service,MacUpdateService)==(platform=='darwin')
        assert bridge.service.root==tmp_path/'cache'
        assert bridge.service.state_root==tmp_path/'state'
    finally:bridge.close()


def test_mac_update_service_keeps_separate_state_and_never_contacts_windows_channels(tmp_path):
    state_root=tmp_path/'preferences';state_root.mkdir()
    (state_root/'state.json').write_text(json.dumps({'autoCheck':False}),encoding='utf8')
    class NoNetwork:
        def __getattr__(self,name):
            raise AssertionError('Mac installer updates must not use network transport')
    service=MacUpdateService(data_root=tmp_path/'cache',state_root=state_root,transport=NoNetwork())
    assert service.root==tmp_path/'cache' and service.state_root==state_root
    assert service.preferences()['autoCheck'] is False
    service.start_maintenance()
    assert service._maintenance_thread is None
    for action in (service.check,service.download,service.apply):
        with pytest.raises(UpdateError) as error:action()
        assert error.value.code=='PLATFORM_UNAVAILABLE'
    assert not service.root.exists()


@pytest.mark.parametrize('platform',['win32','darwin'])
def test_m05_uses_platform_process_and_preserves_playback_source(tmp_path,monkeypatch,platform):
    from ptb_worker import m05_media_export
    calls=[]
    class Process:
        def __init__(self,args,root,limit):
            calls.append((args,root,limit))
            (root/'export-response.json').write_text(json.dumps({'ok':True,'value':{'ready':True}}))
        def poll(self):return 0
        def close(self):calls.append('closed')
    monkeypatch.setitem(sys.modules,'ptb_worker.native.'+('darwin' if platform=='darwin' else 'windows'),SimpleNamespace(OwnedProcess=Process))
    monkeypatch.setenv('PTB_M05_PYTHON',str(tmp_path/'windows-python.exe'))
    (tmp_path/'windows-python.exe').touch()
    monkeypatch.setattr(sys,'platform',platform)
    source=tmp_path/'source.json';source.write_text('{}')
    assert M05Bridge.export_recording(source,action='playback')=={'ready':True}
    args,root,limit=calls[0]
    assert args[0]==(sys.executable if platform=='darwin' else str(tmp_path/'windows-python.exe'))
    assert args[1:4]==['-I','-B',str(Path(m05_media_export.__file__))]
    assert root==tmp_path and limit==1_073_741_824 and calls[-1]=='closed'
    assert source.exists()


@pytest.mark.parametrize('platform',['win32','darwin'])
def test_recording_supported_platform_reaches_device_validation(monkeypatch,platform):
    monkeypatch.setattr(sys,'platform',platform)
    monkeypatch.setattr(recording,'devices',lambda backend:[{'id':'owned-test'}])
    service=recording.RecordingService()
    assert service.dispatch({'op':'capabilities'})['devices']==[{'id':'owned-test'}]
    monkeypatch.setattr(service,'require',lambda **options:None)
    class DeviceValidated(Exception):pass
    def validate(config,backend):raise DeviceValidated()
    monkeypatch.setattr(recording,'validate_config',validate)
    for action in ('start','probe_start'):
        with pytest.raises(DeviceValidated):service.dispatch({'op':action,'config':{}})


def test_unsupported_recording_platform_remains_blocked(monkeypatch):
    monkeypatch.setattr(sys,'platform','linux')
    service=recording.RecordingService()
    assert service.dispatch({'op':'capabilities'})['available'] is False
    with pytest.raises(ValueError,match='尚未开放'):service.dispatch({'op':'start'})
