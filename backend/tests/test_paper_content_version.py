"""Local reader versions use existing managed hashes without reading all PDFs."""
import os
from pathlib import Path

from ptb_api.papers import library_content_versions


def test_managed_local_version_is_stable_across_request_hardlinks(tmp_path, monkeypatch):
    asset = tmp_path / 'immutable.pdf'; asset.write_bytes(b'%PDF-fixture')
    first = tmp_path / 'request-a'; second = tmp_path / 'request-b'
    first.mkdir(); second.mkdir()
    os.link(asset, first / 'same.pdf'); os.link(asset, second / 'same.pdf')
    stat = asset.stat(); known = {(stat.st_dev, stat.st_ino): 'verified-content-sha'}
    def no_read(*args, **kwargs):
        raise AssertionError('metadata refresh must not read PDF contents')
    monkeypatch.setattr(Path, 'open', no_read)
    values = []
    for root in (first, second):
        value = {'root': str(root), 'documents': [{'relative': 'same.pdf', 'contentVersion': 'temporary-stat'}]}
        library_content_versions(value, known); values.append(value['documents'][0]['contentVersion'])
    assert values == ['sha256:verified-content-sha'] * 2
