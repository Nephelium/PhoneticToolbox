"""Frozen .app data symlinks and validation-receipt drift boundaries."""
import hashlib,json,sys
from pathlib import Path
import pytest
from ptb_worker.native import darwin
pytestmark=pytest.mark.skipif(sys.platform!='darwin',reason='macOS bundle layout')

def test_frozen_resource_symlink_and_tampered_bytes(tmp_path,monkeypatch):
    content=tmp_path/'Contents';frameworks=content/'Frameworks';resources=content/'Resources'
    frameworks.mkdir(parents=True);resources.mkdir();(resources/'backend').mkdir();(resources/'resources').mkdir()
    (frameworks/'backend').symlink_to('../Resources/backend');(frameworks/'resources').symlink_to('../Resources/resources')
    source=resources/'backend/example.py';source.write_text('value=1\n')
    receipt=dict(schema='ptb-macos-validation/1',versions=darwin.VERSIONS,operations={'acoustic_analysis':True},source_sha256={'backend/example.py':hashlib.sha256(source.read_bytes()).hexdigest()})
    (resources/'resources/macos-validation.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(darwin,'bundle_root',lambda:frameworks);monkeypatch.setattr(sys,'frozen',True,raising=False)
    darwin._validation.cache_clear()
    try:
        assert darwin.capability('acoustic_analysis')
        source.write_text('value=2\n');darwin._validation.cache_clear();assert not darwin.capability('acoustic_analysis')
    finally:darwin._validation.cache_clear()

def test_receipt_rejects_external_symlink_even_with_matching_hash(tmp_path,monkeypatch):
    root=tmp_path/'snapshot';root.mkdir();(root/'resources').mkdir();outside=tmp_path/'external.py';outside.write_text('external')
    (root/'source.py').symlink_to(outside)
    receipt=dict(schema='ptb-macos-validation/1',versions=darwin.VERSIONS,operations={'acoustic_analysis':True},source_sha256={'source.py':hashlib.sha256(outside.read_bytes()).hexdigest()})
    (root/'resources/macos-validation.json').write_text(json.dumps(receipt));monkeypatch.setattr(darwin,'bundle_root',lambda:root)
    darwin._validation.cache_clear()
    try:assert not darwin.capability('acoustic_analysis')
    finally:darwin._validation.cache_clear()
