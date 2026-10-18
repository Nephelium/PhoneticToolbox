"""Explicit shader-cache cleanup after shared browser profiles stop using it."""
import json
from . import cache_location as location
from .platform_paths import user_data_root, windows_workbench_storage
from .startup_cache import FileLock, active_leases, atomic_json, lease, plain, remove_owned

NAMES = ('GPUCache', 'DawnGraphiteCache', 'DawnWebGPUCache', 'ShaderCache', 'GrShaderCache', 'Code Cache')


def profile_lease():
    return lease(user_data_root() / 'browser-cache-usage')


def request_clear(name):
    if name not in NAMES:
        raise ValueError('不支持的图形缓存目录。')
    root = user_data_root()
    request = plain(root / 'graphics-cache-clear.json', root)
    with FileLock(root / '.graphics-cache.lock'):
        names = []
        if request.is_file():
            value = json.loads(request.read_text('utf8'))
            if not isinstance(value,dict) or value.get('schema') != 'ptb-graphics-cache-clear/1' or not isinstance(value.get('names'), list):
                raise ValueError('图形缓存清理记录无法读取，请重新检查。')
            names = value['names']
        if any(item not in NAMES for item in names):
            raise ValueError('图形缓存清理记录无法确认，请重新检查。')
        atomic_json(request, {'schema': 'ptb-graphics-cache-clear/1', 'names': list(dict.fromkeys([*names, name]))})


def cleanup_requested():
    root = user_data_root()
    request = plain(root / 'graphics-cache-clear.json', root)
    if not request.is_file():
        return {'complete': True, 'requested': False}
    # A user has explicitly confirmed this request. Failures retain both the
    # directory and request for a later launch/last-window exit to retry.
    try:
        with FileLock(root / '.graphics-cache.lock'):
            if active_leases(root / 'browser-cache-usage'):
                return {'complete': False, 'active': True}
            from .legacy_cache import _active
            if _active(root / 'startup-cache'):
                return {'complete': False, 'active': True}
            if request.stat().st_size > 4096:
                raise ValueError('Invalid graphics cleanup request')
            value = json.loads(request.read_text('utf8'))
            if not isinstance(value,dict) or value.get('schema') != 'ptb-graphics-cache-clear/1' or not isinstance(value.get('names'), list) or any(name not in NAMES for name in value['names']):
                raise ValueError('Invalid graphics cleanup request')
            retained = []
            browser = windows_workbench_storage()
            for name in value['names']:
                try:
                    path = plain(browser / name, browser)
                    if path.exists():
                        remove_owned(path, browser)
                except (OSError, ValueError):
                    retained.append(name)
            if retained:
                atomic_json(request, {'schema': 'ptb-graphics-cache-clear/1', 'names': retained})
            else:
                request.unlink()
            result = {'complete': not retained, 'retained': retained}
            atomic_json(root / 'graphics-cache-cleanup-report.json', result)
            return result
    except (OSError, ValueError):
        return {'complete': False, 'retained': True}
