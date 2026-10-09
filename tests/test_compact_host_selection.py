"""Sharing requires identical bytes and a bound, intact scientific payload."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('ptb_release_compact_host', Path(__file__).resolve().parents[1]/'release/compact_host.py')
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)


def row(raw):
    return dict(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def stage(tmp_path):
    archive=tmp_path/'runtime-archive';archive.mkdir()
    (archive/'runtime-payload.tar.xz').write_bytes(b'fixture opaque archive')
    files={'runtimes/m05/other-name.dll':row(b'same'), 'runtimes/m05/same-name.dll':row(b'different')}
    (archive/'runtime-files.json').write_text(json.dumps(dict(schema='ptb-runtime-files/1',files=files)),encoding='utf8')
    (tmp_path/'desktop-bundle.json').write_text(json.dumps(dict(runtimeArchive=dict(
        sha256=PACK.digest(archive/'runtime-payload.tar.xz'),manifestSha256=PACK.digest(archive/'runtime-files.json')))),encoding='utf8')
    return archive


def test_content_identity_not_filename_controls_sharing(tmp_path):
    archive=stage(tmp_path)
    records={'same-name.dll':row(b'same'),'unrelated.dll':row(b'host only')}
    value=PACK.shared_manifest(records,archive)
    assert value['files']==records
    assert value['sharedFiles']=={'same-name.dll':'runtimes/m05/other-name.dll'}
    assert PACK.shared_manifest(records,None)==dict(schema='ptb-host-files/1',files=records)


@pytest.mark.parametrize('filename',['runtime-payload.tar.xz','runtime-files.json'])
def test_changed_science_payload_is_rejected(tmp_path,filename):
    archive=stage(tmp_path)
    with (archive/filename).open('ab') as output:output.write(b'changed')
    with pytest.raises(ValueError,match='identity changed'):
        PACK.shared_manifest({'same-name.dll':row(b'same')},archive)
