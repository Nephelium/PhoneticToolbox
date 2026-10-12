import os
from pathlib import Path
import sys
import pytest
from ptb_worker import native_runtime as runtime

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows runtime junctions')


@pytest.fixture(autouse=True)
def cleanup():
    yield
    runtime.close_runtime_entries()


def test_unicode_alias_preserves_files_and_cleans_only_entry(tmp_path):
    original = tmp_path / '中文运行库' / 'python.exe'
    original.parent.mkdir()
    original.write_bytes(b'unchanged runtime')
    alias = Path(runtime.runtime_entry(original))
    assert str(alias).isascii() and len(str(alias)) < 140
    assert alias.samefile(original)
    assert runtime.runtime_entry(original) == str(alias)
    assert runtime.checked_runtime(alias) == alias
    runtime.close_runtime_entries()
    assert not alias.parent.parent.exists()
    assert original.read_bytes() == b'unchanged runtime'


def test_unregistered_and_retargeted_aliases_are_rejected(tmp_path):
    import _winapi
    original = tmp_path / '中文运行库' / 'python.exe'
    original.parent.mkdir()
    original.write_bytes(b'original')
    other = tmp_path / 'other'
    other.mkdir()
    (other / original.name).write_bytes(b'other')
    unregistered = tmp_path / 'unregistered'
    _winapi.CreateJunction(str(other), str(unregistered))
    try:
        with pytest.raises(ValueError, match='Reparse'):
            runtime.checked_runtime(unregistered / original.name)
    finally:
        os.rmdir(unregistered)
    alias = Path(runtime.runtime_entry(original))
    os.rmdir(alias.parent)
    _winapi.CreateJunction(str(other), str(alias.parent))
    with pytest.raises(ValueError, match='target changed'):
        runtime.checked_runtime(alias)
    runtime.close_runtime_entries()
    assert original.is_file() and (other / original.name).is_file()


def test_unicode_temp_uses_private_ascii_fallback(tmp_path, monkeypatch):
    original = tmp_path / '中文运行库' / 'python.exe'
    original.parent.mkdir()
    original.write_bytes(b'original')
    monkeypatch.setattr(runtime.tempfile, 'gettempdir', lambda: str(tmp_path / '中文临时目录'))
    alias = Path(runtime.runtime_entry(original))
    assert alias.is_relative_to(Path(os.environ['SystemRoot']) / 'Temp')
    assert str(alias).isascii() and alias.samefile(original)
