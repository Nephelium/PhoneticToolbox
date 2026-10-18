import json

import pytest

from ptb_desktop import graphics_cache as graphics
from ptb_desktop.platform_paths import user_data_root, windows_workbench_storage


@pytest.fixture
def browser(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'profile'))
    root = windows_workbench_storage()
    (root / 'GPUCache').mkdir(parents=True)
    (root / 'GPUCache/data').write_bytes(b'cache')
    (root / 'IndexedDB').mkdir()
    (root / 'IndexedDB/drafts').write_bytes(b'research-draft')
    return root


def test_graphics_cleanup_waits_for_last_profile_then_preserves_drafts(browser):
    live = graphics.profile_lease()
    graphics.request_clear('GPUCache')
    try:
        assert graphics.cleanup_requested()['active']
        assert (browser / 'GPUCache/data').exists()
    finally:
        live.close()
    assert graphics.cleanup_requested()['complete']
    assert not (browser / 'GPUCache').exists()
    assert (browser / 'IndexedDB/drafts').read_bytes() == b'research-draft'


def test_no_request_never_removes_graphics_cache(browser):
    assert graphics.cleanup_requested()['requested'] is False
    assert (browser / 'GPUCache/data').exists()


def test_arbitrary_name_or_tampered_request_is_retained(browser):
    with pytest.raises(ValueError):graphics.request_clear('IndexedDB')
    request = user_data_root() / 'graphics-cache-clear.json'
    request.parent.mkdir(parents=True, exist_ok=True)
    request.write_text(json.dumps({'schema':'ptb-graphics-cache-clear/1','names':['IndexedDB']}), encoding='utf8')
    assert not graphics.cleanup_requested()['complete']
    assert (browser / 'IndexedDB/drafts').exists()
    request.write_text('[]', encoding='utf8')
    assert not graphics.cleanup_requested()['complete']


def test_delete_failure_keeps_request_for_later_retry(browser, monkeypatch):
    graphics.request_clear('GPUCache')
    remove = graphics.remove_owned
    def denied(*args):raise PermissionError('in use')
    monkeypatch.setattr(graphics, 'remove_owned', denied)
    assert not graphics.cleanup_requested()['complete']
    assert (user_data_root() / 'graphics-cache-clear.json').exists()
    assert (browser / 'GPUCache/data').exists()
    monkeypatch.setattr(graphics, 'remove_owned', remove)
    assert graphics.cleanup_requested()['complete']
