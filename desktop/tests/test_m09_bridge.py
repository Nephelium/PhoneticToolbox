import pytest
from ptb_desktop.m09_bridge import M09Bridge
from ptb_desktop.file_provider import FileAccessError


def test_python_check_requires_native_picker_grant(tmp_path,monkeypatch):
    from ptb_worker import m09_runtime
    bridge=M09Bridge()
    with pytest.raises(FileAccessError,match='先选择'):bridge.check(str(tmp_path/'python.exe'))
    python=tmp_path/'python.exe';python.write_bytes(b'owned fixture')
    grant=bridge.grant(python)
    monkeypatch.setattr(m09_runtime,'check_and_register',lambda p:dict(cuda=True))
    assert bridge.check(grant['id'])==dict(cuda=True)
    python.write_bytes(b'changed')
    with pytest.raises(FileAccessError,match='已改变'):bridge.check(grant['id'])
