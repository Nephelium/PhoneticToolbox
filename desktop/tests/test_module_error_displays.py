"""Product errors keep their raw identity and an explicit display descriptor."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ptb_desktop.file_provider import FileAccessError
from ptb_desktop.native_i18n import NativeLocale,native_error_payload
from ptb_desktop.m05_bridge import M05Bridge
from ptb_desktop.m06_bridge import M06Bridge
from ptb_desktop.m08_bridge import M08Bridge
from ptb_desktop.m09_bridge import M09Bridge,MESSAGES,MESSAGE_KEYS


def translated(error,key):
    raw=str(error);payload=json.loads(json.dumps(native_error_payload(error),ensure_ascii=False))
    assert payload['error']==raw
    assert payload['display']=={'key':'native.'+key,'values':{}}
    language=NativeLocale();texts=[]
    for code in ('zh-CN','en','ja'):
        language.select(code);texts.append(language.text(key))
        assert str(error)==raw
    assert len(set(texts))==3


@pytest.mark.parametrize('action,timeout,key',[
    ('playback',True,'playbackTimeout'),('export',True,'recordingSaveTimeout'),
    ('playback',False,'playbackIncomplete'),('export',False,'recordingEncodingIncomplete'),
])
def test_actual_m05_export_rejections_keep_recording_and_display(tmp_path,monkeypatch,action,timeout,key):
    source=tmp_path/'被试甲.webm';source.write_bytes(b'owned recorder fixture')
    class Process:
        def __init__(self,*args):pass
        def poll(self):return None if timeout else 1
        def close(self):pass
    monkeypatch.setenv('PTB_M05_PYTHON',sys.executable)
    with patch('ptb_worker.native.windows.OwnedProcess',Process),patch('ptb_desktop.m05_bridge.time.monotonic',side_effect=[0,621]):
        with pytest.raises(FileAccessError) as caught:M05Bridge.export_recording(source,action=action)
    translated(caught.value,'errors.m05.'+key);assert source.read_bytes()==b'owned recorder fixture'


@pytest.mark.parametrize('code',list(MESSAGES))
def test_actual_m09_fingerprint_failure_keeps_code_mapping(tmp_path,monkeypatch,code):
    from ptb_worker import m09_runtime
    source=tmp_path/'python.exe';source.write_bytes(b'owned environment fixture')
    def rejected(path):raise ValueError(code)
    monkeypatch.setattr(m09_runtime,'fingerprint',rejected)
    with pytest.raises(FileAccessError) as caught:M09Bridge().grant(source)
    assert str(caught.value)==MESSAGES[code];translated(caught.value,MESSAGE_KEYS[code])
    assert source.read_bytes()==b'owned environment fixture'


def test_m06_and_m08_early_rejections_keep_original_protocol_text():
    with pytest.raises(FileAccessError) as caught:M06Bridge(None).invoke({'op':'m06_parameters','text':'x'*2_000_001})
    assert str(caught.value)=='完整参数超过 2 MB';translated(caught.value,'errors.m06.parameterSize')
    job={'operation':'pitch_manipulation','state':'succeeded','result_manifest':{'files':[]}}
    bridge=M08Bridge(SimpleNamespace(provider=SimpleNamespace(directory=lambda key:SimpleNamespace(purpose='output')),service=SimpleNamespace(get=lambda path:job)))
    with pytest.raises(FileAccessError) as caught:bridge.export({'directory':'owned','job':'00000000-0000-4000-8000-000000000001','id':'missing'})
    assert str(caught.value)=='结果不可用。';translated(caught.value,'errors.m08.resultUnavailable')


def test_unknown_native_diagnostic_is_not_reinterpreted():
    raw='ExternalCodec: 被试甲.webm E_RAW_17';payload=native_error_payload(RuntimeError(raw))
    assert payload=={'ok':False,'error':raw}
