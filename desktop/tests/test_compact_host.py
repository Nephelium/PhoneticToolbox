from io import BytesIO
import hashlib
import json
import tarfile

import pytest
from ptb_desktop.compact_host import expand


def fixture(root, *, name='PyQt6/QtCore.pyd', body=b'original-Qt', expected=None):
    path = root / 'host-payload.tar.xz'
    with tarfile.open(path, 'w:xz') as archive:
        info = tarfile.TarInfo(name); info.size = len(body)
        archive.addfile(info, BytesIO(body))
    files = {name: dict(size=len(body), sha256=hashlib.sha256(body if expected is None else expected).hexdigest())}
    manifest = root / 'host-files.json'
    manifest.write_text(json.dumps(dict(schema='ptb-host-files/1', files=files)), 'utf8')
    (root / 'host-archive.json').write_text(json.dumps(dict(schema='ptb-host-archive/1',
        size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        manifestSha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        expandedBytes=len(body), files=1)), 'utf8')


def test_original_host_bytes_and_child_reuse(tmp_path):
    fixture(tmp_path)
    expand(tmp_path); expand(tmp_path)
    assert (tmp_path / 'PyQt6/QtCore.pyd').read_bytes() == b'original-Qt'


@pytest.mark.parametrize('name', ['../escape.dll', '/absolute', 'Qt6Core.dll:ads', 'PyQt6/CON.dll'])
def test_host_unsafe_target_rejected(tmp_path, name):
    fixture(tmp_path, name=name)
    with pytest.raises(ValueError): expand(tmp_path)
    assert not (tmp_path / '.ptb-host-ready.json').exists()


@pytest.mark.parametrize('damage', ['member', 'collision', 'archive'])
def test_host_corruption_or_bootstrap_collision_never_completes(tmp_path, damage):
    fixture(tmp_path, expected=b'bad' if damage == 'member' else None)
    if damage == 'collision':
        existing = tmp_path / 'PyQt6/QtCore.pyd'; existing.parent.mkdir(); existing.write_bytes(b'bootstrap')
    elif damage == 'archive':
        (tmp_path / 'host-payload.tar.xz').write_bytes(b'corrupt')
    with pytest.raises(ValueError): expand(tmp_path)
    assert not (tmp_path / '.ptb-host-ready.json').exists()
    if damage == 'collision': assert existing.read_bytes() == b'bootstrap'
