"""Boundary provenance stays bound to geometry/associations, separately from labels."""
import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('package_geolinguistics_data',Path(__file__).parents[1]/'scripts/package_geolinguistics_data.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source(tmp_path):
    for name in module.DATA_FILES:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixed '+name.encode())
    manifest=module.write_data_manifest(tmp_path)
    manifest['boundaries']={'schema':'ptb-geolinguistics-boundary-provenance/1','version':'synthetic-boundary-test','status':'candidate','sources':[{'sourceId':'test-source','license':'test-license'}]}
    manifest['distribution']='candidate; publication pending'
    (tmp_path/'data-manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    return manifest


def test_reviewed_labels_refresh_keeps_boundary_provenance(tmp_path):
    previous=source(tmp_path)
    (tmp_path/'catalog.json').write_bytes(b'new reviewed labels')
    result=module.write_data_manifest(tmp_path)
    assert result['boundaries']==previous['boundaries']
    assert result['distribution']==previous['distribution']
    assert result['files'][0]['sha256']!=previous['files'][0]['sha256']


@pytest.mark.parametrize('name',['maps/province.json','sites.json','classification.json','map-index.json'])
def test_changed_geometry_or_association_cannot_inherit_stale_provenance(tmp_path,name):
    source(tmp_path)
    (tmp_path/name).write_bytes(b'changed')
    with pytest.raises(ValueError,match='provenance'):
        module.write_data_manifest(tmp_path)


def test_legacy_trial_does_not_acquire_public_osm_provenance(tmp_path):
    for name in module.DATA_FILES:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'legacy')
    assert 'boundaries' not in module.write_data_manifest(tmp_path)


def test_packaging_preserves_reviewed_manifest_exact_bytes(tmp_path):
    original=tmp_path/'input'
    original.mkdir()
    manifest=source(original)
    raw=(json.dumps(manifest,ensure_ascii=False,indent=4)+'\r\n').encode('utf-8')
    (original/'data-manifest.json').write_bytes(raw)
    destination=tmp_path/'output'
    module.package_data(original,destination)
    assert (destination/'data-manifest.json').read_bytes()==raw
