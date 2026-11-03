"""The actual freeze must retain download metadata while excluding foreign ZIPs."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_v3_local_preview import copy_runtime_frontend


def reader(tmp_path):
    source = tmp_path / 'frontend'
    manual = source / 'manual'
    (manual / 'chapters').mkdir(parents=True)
    (manual / 'assets').mkdir()
    (source / 'index.html').write_text('owned fixture', encoding='utf-8')
    image = b'owned source media'
    (manual / 'assets/source.png').write_bytes(image)
    (manual / 'chapters/introduction.json').write_text(json.dumps({'body': {'type': 'doc', 'content': [
        {'type': 'image', 'attrs': {'assetId': 'source'}}]}}), encoding='utf-8')
    (manual / 'project.json').write_text(json.dumps({'chapters': [{'path': 'chapters/introduction.json'}],
        'assets': [{'id': 'source', 'path': 'assets/source.png', 'sha256': hashlib.sha256(image).hexdigest()}]}), encoding='utf-8')
    manifest = {'schemaVersion': 'ptb-manual-downloads/1', 'published': False, 'packs': []}
    (manual / 'language-packs.json').write_text(json.dumps(manifest), encoding='utf-8')
    # Unexpected old files in the generated manual must never be frozen by a copy-all.
    (manual / 'unreferenced-en.zip').write_bytes(b'foreign fixture')
    return source


def test_real_frontend_freeze_keeps_manual_download_manifest_exactly(tmp_path):
    source = reader(tmp_path)
    target = tmp_path / 'frozen'
    copy_runtime_frontend(source, target, tmp_path / 'audit.json')
    assert (target / 'manual/language-packs.json').read_bytes() == (source / 'manual/language-packs.json').read_bytes()
    assert (target / 'manual/assets/source.png').read_bytes() == b'owned source media'
    assert not list(target.rglob('*.zip'))


def test_stale_reader_without_download_metadata_is_not_a_complete_freeze(tmp_path):
    source = reader(tmp_path)
    (source / 'manual/language-packs.json').unlink()
    with pytest.raises(FileNotFoundError):
        copy_runtime_frontend(source, tmp_path / 'frozen', tmp_path / 'audit.json')


def test_identical_chinese_media_is_encoded_and_stored_once_without_merging_asset_ids(tmp_path, monkeypatch):
    from PIL import Image
    import sys
    sys.path.insert(0, str(ROOT / 'scripts/manual'))
    import pack_media
    source = reader(tmp_path)
    image = source / 'manual/assets/source.png'
    Image.new('RGB', (128, 128), '#234567').save(image)
    project_file = source / 'manual/project.json'
    project = json.loads(project_file.read_text('utf-8'))
    first = project['assets'][0]
    first.update(kind='image', sha256=hashlib.sha256(image.read_bytes()).hexdigest())
    project['assets'].append({**first, 'id': 'source-copy', 'caption': 'Independent caption'})
    project_file.write_text(json.dumps(project), encoding='utf-8')
    chapter = source / 'manual/chapters/introduction.json'
    content = json.loads(chapter.read_text('utf-8'))
    content['body']['content'].append({'type': 'image', 'attrs': {'assetId': 'source-copy'}})
    chapter.write_text(json.dumps(content), encoding='utf-8')
    calls = []
    original = pack_media.package_asset
    def encode(*args, **kwargs):
        calls.append(args[0]['id'])
        return original(*args, **kwargs)
    monkeypatch.setattr(pack_media, 'package_asset', encode)
    target = tmp_path / 'frozen'
    copy_runtime_frontend(source, target, tmp_path / 'audit.json', optimize_media=True)
    packed = json.loads((target / 'manual/project.json').read_text('utf-8'))['assets']
    assert calls == ['source']
    assert [row['id'] for row in packed] == ['source', 'source-copy']
    assert packed[1]['caption'] == 'Independent caption'
    assert packed[0]['path'] == packed[1]['path']
    with Image.open(target / 'manual' / packed[0]['path']) as result:
        assert result.getpixel((64, 64)) == (35, 69, 103)
    audit = json.loads((tmp_path / 'audit.json').read_text('utf-8'))
    assert audit['uniqueMediaFiles'] == 1 and audit['deduplicatedAssets'] == 1
    assert audit['includedBytes'] == (target / 'manual' / packed[0]['path']).stat().st_size
