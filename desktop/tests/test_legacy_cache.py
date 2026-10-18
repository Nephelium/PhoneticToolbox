import json
from pathlib import Path

import pytest

from ptb_desktop import cache_location as location
from ptb_desktop import legacy_cache
from ptb_desktop.startup_cache import SCHEMA, lease


@pytest.fixture
def caches(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'profile'))
    monkeypatch.delenv(location.ENVIRONMENT, raising=False)
    monkeypatch.delenv(location.MODE_ENVIRONMENT, raising=False)
    location.configure_launcher(tmp_path / 'app.exe', [], lambda reason: tmp_path / 'chosen')
    old = location.configuration_path().parent / 'startup-cache'
    old.mkdir(parents=True)
    (old / 'owner.json').write_text(json.dumps({'schema': SCHEMA}), encoding='utf8')
    (old / 'apps').mkdir()
    (old / 'apps' / 'old.bin').write_bytes(b'old-cache')
    return old


def test_scan_finds_known_old_location_and_never_includes_current_or_research(caches, tmp_path):
    research = location.configuration_path().parent / 'papers'
    research.mkdir()
    (research / 'paper.pdf').write_bytes(b'research')
    candidate = legacy_cache.inspect_candidates()['candidates'][0]
    assert Path(candidate['path']) == caches and candidate['bytes'] > 0 and candidate['can_clean']
    with pytest.raises(ValueError, match='亲自核对'):
        legacy_cache.clean_candidate(candidate['id'], False)
    assert (caches / 'apps' / 'old.bin').exists()
    assert legacy_cache.clean_candidate(candidate['id'], True)['complete']
    assert not caches.exists() and (research / 'paper.pdf').read_bytes() == b'research'
    assert location.portable_directory().is_dir() and location.configuration_path().exists()


def test_active_lease_is_reported_and_cleanup_rechecks_new_lease(caches):
    candidate = legacy_cache.inspect_candidates()['candidates'][0]
    live = lease(caches)
    try:
        current = legacy_cache.inspect_candidates()['candidates'][0]
        assert not current['can_clean'] and '仍有程序' in current['reason']
        with pytest.raises(ValueError, match='仍有程序'):
            legacy_cache.clean_candidate(candidate['id'], True)
        assert (caches / 'apps' / 'old.bin').exists()
    finally:
        live.close()
    assert legacy_cache.clean_candidate(candidate['id'], True)['complete']


def test_unconfirmed_owner_and_unknown_content_are_retained(caches):
    candidate = legacy_cache.inspect_candidates()['candidates'][0]
    (caches / 'owner.json').write_text('{}', encoding='utf8')
    assert not legacy_cache.inspect_candidates()['candidates'][0]['can_clean']
    with pytest.raises(ValueError, match='ownership'):
        legacy_cache.clean_candidate(candidate['id'], True)
    (caches / 'owner.json').write_text(json.dumps({'schema': SCHEMA}), encoding='utf8')
    (caches / 'research.wav').write_bytes(b'keep')
    assert not legacy_cache.clean_candidate(candidate['id'], True)['complete']
    assert (caches / 'research.wav').read_bytes() == b'keep'


def test_old_update_download_cleanup_preserves_update_preferences(caches):
    root = location.configuration_path().parent / 'updates'
    download = root / 'downloads' / ('a' * 32)
    download.mkdir(parents=True)
    (download / 'old.zip').write_bytes(b'old-update')
    state = root / 'state.json'
    state.write_text('{"autoCheck":false}', encoding='utf8')
    candidate = next(row for row in legacy_cache.inspect_candidates()['candidates'] if row['kind'] == 'updates')
    assert legacy_cache.clean_candidate(candidate['id'], True)['complete']
    assert not download.exists() and json.loads(state.read_text('utf8')) == {'autoCheck': False}


def test_arbitrary_directory_identity_is_rejected(caches, tmp_path):
    with pytest.raises(ValueError, match='位置已变化'):
        legacy_cache.clean_candidate(str(tmp_path), True)


def test_current_directory_cannot_be_cleaned_after_location_changes(caches, monkeypatch):
    candidate = legacy_cache.inspect_candidates()['candidates'][0]
    monkeypatch.delenv(location.ENVIRONMENT)
    with pytest.raises(ValueError, match='位置已变化'):
        legacy_cache.clean_candidate(candidate['id'], True)
    assert (caches / 'apps' / 'old.bin').exists()
