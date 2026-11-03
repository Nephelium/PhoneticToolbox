"""Owned QA cache discovery uses the production safety/cleanup algorithms."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from ptb_desktop import legacy_cache
from ptb_desktop.startup_cache import SCHEMA, lease
from ptb_desktop.task_bridge import TaskBridge
from ptb_desktop.host import Bridge


def populated_context(tmp_path):
    context = legacy_cache.isolated_context(tmp_path / 'owned-qa')
    old = context.data_root / 'startup-cache'
    (old / 'apps').mkdir(parents=True)
    (old / 'owner.json').write_text(json.dumps({'schema': SCHEMA}), encoding='utf8')
    (old / 'apps' / 'old.bin').write_bytes(b'owned-cache')
    return context, old


def test_context_discovery_and_cleanup_never_resolve_personal_locations(tmp_path):
    context, old = populated_context(tmp_path)
    outside = tmp_path / 'personal-sentinel'
    outside.mkdir()
    sentinel = outside / 'research.wav'
    sentinel.write_bytes(b'keep')
    with patch.object(legacy_cache, 'user_data_root', side_effect=AssertionError('Personal root read')), \
         patch.object(legacy_cache, 'windows_workbench_storage', side_effect=AssertionError('Personal browser read')), \
         patch.object(legacy_cache.location, 'previous_directories', side_effect=AssertionError('Personal preferences read')), \
         patch.object(legacy_cache.location, 'portable_directory', side_effect=AssertionError('Global cache read')):
        status = legacy_cache.cache_location(context)
        assert Path(status['path']).is_relative_to(context.owned_root)
        assert status['exit_cleanup_supported'] is False
        candidate = legacy_cache.inspect_candidates(context)['candidates'][0]
        assert Path(candidate['path']) == old
        assert candidate['label_key'] == 'cache.candidate.runtime'
        assert candidate['reason_code'] == '' and candidate['can_clean']
        with pytest.raises(ValueError, match='亲自核对'):
            legacy_cache.clean_candidate(candidate['id'], False, context)
        with patch.object(legacy_cache, '_open', return_value={'opened': True}) as opened:
            assert legacy_cache.open_candidate(candidate['id'], context)['opened']
            opened.assert_called_once_with(old)
        assert legacy_cache.clean_candidate(candidate['id'], True, context)['message_code'] == 'complete'
    assert not old.exists() and sentinel.read_bytes() == b'keep'


def test_context_keeps_lease_and_unknown_owner_safety(tmp_path):
    context, old = populated_context(tmp_path)
    key = legacy_cache.inspect_candidates(context)['candidates'][0]['id']
    live = lease(old)
    try:
        candidate = legacy_cache.inspect_candidates(context)['candidates'][0]
        assert candidate['reason_code'] == 'in_use' and not candidate['can_clean']
        with pytest.raises(ValueError, match='仍有程序'):
            legacy_cache.clean_candidate(key, True, context)
        assert (old / 'apps' / 'old.bin').exists()
    finally:
        live.close()
    (old / 'owner.json').write_text('{}', encoding='utf8')
    assert legacy_cache.inspect_candidates(context)['candidates'][0]['reason_code'] == 'ownership_unknown'
    with pytest.raises(ValueError, match='ownership'):
        legacy_cache.clean_candidate(key, True, context)


def test_context_rejects_external_roots_and_ids(tmp_path):
    context, _ = populated_context(tmp_path)
    with pytest.raises(ValueError, match='escapes'):
        legacy_cache.CacheContext(context.owned_root, tmp_path / 'outside', context.workbench_root, context.current_root)
    with pytest.raises(ValueError, match='位置已变化'):
        legacy_cache.open_candidate(str(tmp_path), context)
    with pytest.raises(ValueError, match='位置已变化'):
        legacy_cache.clean_candidate(str(tmp_path), True, context)


def test_task_bridge_passes_owned_context_to_real_discovery(tmp_path):
    context, old = populated_context(tmp_path)
    bridge = TaskBridge.__new__(TaskBridge)
    bridge.cache_context = context
    candidate = bridge.invoke({'op': 'cache_candidates'})['candidates'][0]
    assert Path(candidate['path']) == old
    assert bridge.invoke({'op': 'clean_cache_candidate', 'id': candidate['id'], 'confirmed': True})['complete']


def test_task_bridge_initializes_manual_packs_only_in_explicit_owned_root(tmp_path):
    import threading
    bridge = TaskBridge.__new__(TaskBridge)
    bridge.manual_pack_root = tmp_path / 'owned-qa' / 'manual-packs'
    bridge._manual_packs = None
    bridge._manual_packs_lock = threading.Lock()
    with patch('ptb_desktop.manual_packs.ManualPackManager') as manager:
        manager.return_value.invoke.return_value = {'installed': []}
        assert bridge.invoke({'op': 'manual_packs', 'action': 'status'}) == {'installed': []}
        manager.assert_called_once_with(root=bridge.manual_pack_root)
        manager.return_value.invoke.assert_called_once_with('status', {})


def test_task_bridge_close_cancels_owned_manual_manager_and_prevents_new_transfers(tmp_path):
    import threading
    from ptb_desktop.file_provider import FileAccessError
    bridge = TaskBridge.__new__(TaskBridge)
    bridge._manual_packs = Mock()
    bridge._manual_packs_lock = threading.Lock()
    bridge.close()
    bridge._manual_packs.close.assert_called_once_with()
    with pytest.raises(FileAccessError, match='manual.pack.cancelled'):
        bridge.invoke({'op': 'manual_packs', 'action': 'importBegin'})


def test_qa_graphics_and_exit_cleanup_are_explicitly_disabled(tmp_path):
    context, _ = populated_context(tmp_path)
    (context.workbench_root / 'GPUCache').mkdir(parents=True)
    graphics = next(row for row in legacy_cache.inspect_candidates(context)['candidates'] if row['kind'] == 'graphics')
    assert not graphics['can_clean'] and graphics['reason_code'] == 'qa_exit_cleanup_disabled'
    coordinator = SimpleNamespace(clear_caches=Mock(side_effect=AssertionError('Production cleanup invoked')))
    owner = SimpleNamespace(window=SimpleNamespace(cache_context=context, update_coordinator=coordinator))
    for op in ('clear_all_caches', 'clear_graphics_cache'):
        reply = json.loads(Bridge.invoke(owner, json.dumps({'op': op, 'purpose': 'confirmed', 'id': graphics['id']})))
        assert reply['ok'] is False
        assert 'native.cache.qaExitCleanupUnavailable' in str(reply)
    coordinator.clear_caches.assert_not_called()
