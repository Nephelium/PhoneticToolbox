import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile

import pytest

from ptb_desktop.compact_runtime import expand, relative_parts
from ptb_desktop import compact_runtime


def payload(root, *, name='runtimes/egg/python.exe', content=b'MZoriginal', recorded=None, link=False):
    archive = root / 'runtime-payload.tar.xz'
    with tarfile.open(archive, 'w:xz') as stream:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        if link:
            info.type = tarfile.SYMTYPE
            info.linkname = 'outside'
        stream.addfile(info, BytesIO(content) if not link else None)
    files = {name: {'size': len(content), 'sha256': hashlib.sha256(content if recorded is None else recorded).hexdigest()}}
    manifest = root / 'runtime-files.json'
    manifest.write_text(json.dumps({'schema': 'ptb-runtime-files/1', 'files': files}), 'utf8')
    descriptor = dict(schema='ptb-runtime-archive/1', archive=archive.name, manifest=manifest.name,
                      size=archive.stat().st_size, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                      manifestSha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      files=1, expandedBytes=len(content))
    (root / 'desktop-bundle.json').write_text(json.dumps({'runtimeArchive': descriptor}), 'utf8')
    return archive


def test_exact_bytes_and_completed_child_bootstrap(tmp_path):
    payload(tmp_path)
    expand(tmp_path)
    assert (tmp_path / 'runtimes/egg/python.exe').read_bytes() == b'MZoriginal'
    expand(tmp_path)
    assert (tmp_path / 'runtimes/.ptb-runtime-complete.json').is_file()


@pytest.mark.parametrize('name', ['runtimes/egg/../escape', 'runtimes/egg/a:b', 'runtimes/egg/CON.txt',
                                  '/runtimes/egg/a', 'runtimes/mfa/python.exe', 'runtimes/egg//a',
                                  'runtimes/egg/a\\b', 'runtimes/egg/a.'])
def test_path_escape_and_windows_aliases(name):
    with pytest.raises(ValueError):
        relative_parts(name)


@pytest.mark.parametrize('damage', ['archive', 'member', 'link'])
def test_corruption_and_links_never_publish_runtime(tmp_path, damage):
    archive = payload(tmp_path, recorded=b'different' if damage == 'member' else None, link=damage == 'link')
    if damage == 'archive':
        archive.write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        expand(tmp_path)
    assert not (tmp_path / 'runtimes').exists()


def test_transient_windows_rename_lock_retries(tmp_path,monkeypatch):
    payload(tmp_path)
    original=Path.rename;attempts=[]
    def locked_once(source,destination):
        attempts.append(source)
        if len(attempts)==1:
            error=PermissionError('owned transient lock');error.winerror=5;raise error
        return original(source,destination)
    monkeypatch.setattr(Path,'rename',locked_once)
    expand(tmp_path)
    assert len(attempts)==2
    assert (tmp_path/'runtimes/egg/python.exe').read_bytes()==b'MZoriginal'


def test_permanent_lock_fails_without_completion(tmp_path,monkeypatch):
    payload(tmp_path)
    clock=iter([0,4]);monkeypatch.setattr(compact_runtime.time,'monotonic',lambda:next(clock))
    def locked(*args):
        error=PermissionError('permanent deny');error.winerror=5;raise error
    monkeypatch.setattr(Path,'rename',locked)
    with pytest.raises(PermissionError):expand(tmp_path)
    assert not (tmp_path/'runtimes/.ptb-runtime-complete.json').exists()


def test_publish_never_overwrites_existing_directory(tmp_path):
    source=tmp_path/'source';source.mkdir()
    target=tmp_path/'target';target.mkdir();(target/'keep').write_bytes(b'keep')
    with pytest.raises(FileExistsError):compact_runtime.publish_directory(source,target)
    assert (target/'keep').read_bytes()==b'keep' and source.exists()
def test_publish_retries_real_windows_directory_lock(tmp_path):
    import ctypes
    from ctypes import wintypes
    import os
    import threading
    import time
    import pytest
    if os.name!='nt':pytest.skip('Windows directory sharing')
    from ptb_desktop.compact_runtime import publish_directory
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateFileW.argtypes=(wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE)
    api.CreateFileW.restype=wintypes.HANDLE;api.CloseHandle.argtypes=(wintypes.HANDLE,)
    source=tmp_path/'staging';source.mkdir();(source/'runtime').write_bytes(b'original')
    handle=api.CreateFileW(str(source),0x80000000,3,None,3,0x02000000,None)
    assert handle not in (None,ctypes.c_void_p(-1).value)
    try:
        with pytest.raises(PermissionError):source.rename(tmp_path/'ready')
    except BaseException:
        api.CloseHandle(handle);raise
    release=threading.Timer(.2,lambda:api.CloseHandle(handle));release.start()
    started=time.monotonic()
    try:publish_directory(source,tmp_path/'ready')
    finally:release.join()
    assert time.monotonic()-started>=.15
    assert (tmp_path/'ready/runtime').read_bytes()==b'original'

