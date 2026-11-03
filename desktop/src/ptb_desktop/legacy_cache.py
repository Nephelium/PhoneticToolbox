"""Bounded discovery and confirmed cleanup of known former cache locations."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time

from . import cache_location as location
from .platform_paths import user_data_root, windows_workbench_storage
from .startup_cache import SCHEMA, FileLock, clear_if_idle, plain, remove_owned


@dataclass(frozen=True)
class CacheContext:
    """Explicit, owned QA cache roots; never changes process-wide locations."""
    owned_root: Path
    data_root: Path
    workbench_root: Path
    current_root: Path
    previous_roots: tuple[Path, ...] = ()

    def checked(self, path):
        root = plain(self.owned_root, Path(self.owned_root).anchor)
        return plain(path, root)

    def __post_init__(self):
        for path in (self.data_root, self.workbench_root, self.current_root, *self.previous_roots):
            self.checked(path)


def isolated_context(root):
    root = plain(Path(root).absolute(), Path(root).absolute().parent)
    current = root / 'current' / 'PhoneticToolbox-cache'
    current.mkdir(parents=True, exist_ok=True)
    return CacheContext(root, root / 'former-user-data', root / 'workbench', current)


def cache_location(context=None):
    if context is None:
        return location.location_status()
    current = context.checked(context.current_root)
    previous = context.checked(context.data_root / 'startup-cache')
    return {'path': str(current), 'custom': True, 'exit_cleanup_supported': False,
            'legacy_path': str(previous) if previous.is_dir() else None}


def open_cache_directory(*, legacy=False, context=None):
    if context is None:
        return location.open_directory(legacy=legacy)
    status = cache_location(context)
    value = status['legacy_path'] if legacy else status['path']
    if value is None:
        raise ValueError('旧缓存目录已不存在。')
    path = context.checked(value)
    if not path.is_dir():
        raise ValueError('缓存目录已不存在，再次启动时会在所选位置重新准备。')
    return _open(path)


def _known(context=None):
    data = user_data_root() if context is None else context.checked(context.data_root)
    workbench = windows_workbench_storage() if context is None else context.checked(context.workbench_root)
    sources = [(data, 'default')]
    previous = location.previous_directories() if context is None else context.previous_roots
    for directory in previous:
        if context is not None:
            directory = context.checked(directory)
        if directory.is_dir() and (directory / 'cache-directory.json').is_file():
            sources.append((directory, 'previous'))
    current = ({'runtime': location.startup_directory(), 'updates': location.update_directory(),
                'browser': location.browser_directory()} if context is None else
               {'runtime': context.checked(context.current_root / 'startup-cache'),
                'updates': context.checked(context.current_root / 'updates'),
                'browser': context.checked(context.current_root / 'browser-cache')})
    rows = []
    for parent, origin in sources:
        paths = {'runtime': parent / 'startup-cache', 'updates': parent / 'updates',
                 'browser': workbench / 'http-cache' if origin == 'default' else parent / 'browser-cache'}
        for kind, path in paths.items():
            if context is not None:
                context.checked(path)
            if path == current[kind] or not path.exists():
                continue
            if kind == 'updates':
                try:
                    plain(path,path.parent)
                    folders=[plain(path/name,path) for name in ('downloads','helpers','apply')]
                    if not any(folder.is_dir() and any(folder.iterdir()) for folder in folders):continue
                except (OSError,ValueError):
                    pass  # Show the unsafe candidate without traversing it.
            key = hashlib.sha256((kind + '|' + str(path)).encode('utf8')).hexdigest()[:32]
            rows.append(dict(id=key, kind=kind, path=path, lease_root=parent / 'startup-cache'))
    from .graphics_cache import NAMES
    for name in NAMES:
        path = workbench / name
        if context is not None:
            context.checked(path)
        if path.exists():
            key = hashlib.sha256(('graphics|' + str(path)).encode('utf8')).hexdigest()[:32]
            rows.append(dict(id=key, kind='graphics', path=path, lease_root=data / 'startup-cache'))
    return list({row['id']: row for row in rows}.values())


class CacheInspectionError(ValueError):
    def __init__(self, code, text):
        self.code = code
        super().__init__(text)


def _active(root):
    """Read-only lease probes: a scan never removes stale lock files."""
    import msvcrt
    root = plain(root, root.parent)
    if not root.exists():
        return False
    marker = plain(root / 'owner.json', root)
    if not marker.is_file() or marker.stat().st_size > 4096 or json.loads(marker.read_text('utf8')) != {'schema': SCHEMA}:
        raise CacheInspectionError('ownership_unknown', '目录归属无法确认，请先打开目录亲自核对。')
    folder = plain(root / 'leases', root)
    if folder.is_dir():
        for path in folder.iterdir():
            plain(path, root)
            if not re.fullmatch('[0-9a-f]{32}\\.lock', path.name):
                raise CacheInspectionError('leases_unknown', '目录中有无法确认的占用记录，请先核对。')
            with path.open('r+b') as stream:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    return True
                else:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    return False


def _size(row, deadline):
    roots = [row['path']]
    if row['kind'] == 'updates':
        roots = [row['path'] / family for family in ('downloads', 'helpers', 'apply')]
    total, count = 0, 0
    for root in roots:
        if not root.exists():
            continue
        plain(root, row['path'].parent)
        for parent, dirs, files in os.walk(root, followlinks=False):
            for name in (*dirs, *files):
                plain(Path(parent) / name, row['path'].parent)
            for name in files:
                total += (Path(parent) / name).stat().st_size
                count += 1
            if count >= 100_000 or time.monotonic() >= deadline:
                return total, False
    return total, True


def inspect_candidates(context=None):
    deadline = time.monotonic() + 3
    result = []
    labels = {'runtime': '旧运行缓存', 'updates': '旧更新下载缓存', 'browser': '旧浏览器缓存', 'graphics': '浏览器图形缓存'}
    for row in _known(context):
        value = dict(id=row['id'], kind=row['kind'], path=str(row['path']), label=labels[row['kind']],
                     label_key='cache.candidate.'+row['kind'], bytes=0, size_complete=False,
                     can_clean=False, clean_on_exit=row['kind']=='graphics', reason='', reason_code='')
        try:
            plain(row['path'], row['path'].parent)
            if row['kind']=='graphics':
                value['can_clean'] = True
                value['reason'] = '核对后可安全退出并清理；仍有其他窗口使用时，等待最后一个窗口退出。'
                value['reason_code'] = 'graphics_exit'
                if context is not None:
                    value['can_clean'] = False
                    value['reason'] = '此验证窗口不执行退出后的缓存清理。'
                    value['reason_code'] = 'qa_exit_cleanup_disabled'
            elif _active(row['lease_root']):
                value['reason'] = '仍有程序使用此处的缓存，请关闭相关窗口后重新检查。'
                value['reason_code'] = 'in_use'
            else:
                value['can_clean'] = True
            value['bytes'], value['size_complete'] = _size(row, deadline)
        except (OSError, ValueError) as error:
            value['can_clean'] = False
            value['reason'] = str(error) if not str(error).isascii() else '目录暂时无法安全检查，已保留。请核对路径和访问权限。'
            value['reason_code'] = error.code if isinstance(error, CacheInspectionError) else 'inspection_failed'
        result.append(value)
    return {'candidates': result}


def _candidate(key, context=None):
    row = next((row for row in _known(context) if row['id'] == key), None)
    if row is None:
        raise ValueError('缓存位置已变化或不存在，请重新检查。')
    plain(row['path'], row['path'].parent)
    return row


def _open(path):
    from PyQt6.QtCore import QUrl
    from PyQt6.QtGui import QDesktopServices
    if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
        raise OSError('目录未能打开，请复制路径后在文件资源管理器中核对。')
    return {'opened': True}


def open_candidate(key, context=None):
    return _open(_candidate(key, context)['path'])


@contextmanager
def _idle(row):
    root = row['lease_root']
    with FileLock(plain(root.parent / '.startup-cache.lock', root.parent)):
        if _active(root):
            raise ValueError('仍有程序使用此处的缓存，请关闭相关窗口后重试。')
        yield


def clean_candidate(key, confirmed, context=None):
    if confirmed is not True:
        raise ValueError('请先打开目录亲自核对，再确认清理。')
    row = _candidate(key, context)
    if row['kind'] == 'graphics':
        raise ValueError('图形缓存需要在窗口退出后清理。')
    if row['kind'] == 'runtime':
        # The existing deletion path rechecks ownership, reparses and leases
        # under the same maintenance lock used by every launcher.
        result = clear_if_idle(row['path'])
        if result.get('active'):
            raise ValueError('仍有程序使用此处的缓存，请关闭相关窗口后重试。')
        complete = result['complete'] and not row['path'].exists()
    else:
        with _idle(row):
            if row['kind'] == 'updates':
                from .update_cache import clean_update_cache
                result = clean_update_cache(row['path'], all_cache=True)
                complete = result['state'] == 'complete' and result['retainedDirectories'] == 0
            else:
                remove_owned(row['path'], row['path'].parent)
                complete = not row['path'].exists()
    return {'complete': complete, 'message_code': 'complete' if complete else 'partial',
            'message': '该处旧缓存已清理。' if complete else '可确认的缓存已清理，仍在使用或无法确认的内容已保留，请打开目录核对。'}
