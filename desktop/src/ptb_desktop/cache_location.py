"""Explicit portable cache location, independent of executable and user data."""
import json
import os
from pathlib import Path
import uuid

from .platform_paths import user_data_root, windows_workbench_storage

ENVIRONMENT = 'PTB_CACHE_DIRECTORY'
MODE_ENVIRONMENT = 'PTB_CACHE_MODE'
DIRECTORY_NAME = 'PhoneticToolbox-cache'
SCHEMA = 'ptb-portable-cache/1'


def _plain(path):
    from .startup_cache import plain
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('请选择完整的缓存目录路径。')
    path = Path(os.path.abspath(path))
    return plain(path, path.anchor)


def distribution_kind(executable):
    marker = Path(executable).absolute().parent / '.ptb-installed.json'
    if not marker.exists():
        return 'portable'
    _plain(marker)
    if marker.stat().st_size > 4096:
        raise ValueError('安装方式标识无效，请重新安装此应用。')
    value = json.loads(marker.read_text('utf8'))
    if not isinstance(value, dict) or value.get('schema') != 'ptb-install/1' or value.get('kind') != 'installer':
        raise ValueError('安装方式标识无效，请重新安装此应用。')
    return 'installer'


def selected_directory(parent):
    parent = _plain(parent)
    return parent if parent.name.casefold() == DIRECTORY_NAME.casefold() else parent / DIRECTORY_NAME


def _validate_directory(path, *, create=False):
    path = _plain(path)
    if path.name.casefold() != DIRECTORY_NAME.casefold():
        raise ValueError(f'请选择专用的 {DIRECTORY_NAME} 缓存文件夹。')
    for protected in (user_data_root(), windows_workbench_storage()):
        protected = protected.absolute()
        if path.is_relative_to(protected) or protected.is_relative_to(path):
            raise ValueError('缓存目录须与设置、草稿及研究资料目录分开。')
    marker = _plain(path / 'cache-directory.json')
    if path.exists():
        if not path.is_dir():
            raise ValueError('缓存位置已被同名文件占用，请重新选择。')
        if marker.exists():
            if marker.stat().st_size > 4096 or json.loads(marker.read_text('utf8')) != {'schema': SCHEMA}:
                raise ValueError('该目录的缓存标识无效，请选择其他文件夹。')
        elif any(path.iterdir()):
            raise ValueError('该文件夹已有其他内容，请选择空文件夹或已有的应用缓存目录。')
    if create:
        from .startup_cache import atomic_json
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ('.write-check-' + uuid.uuid4().hex)
        with probe.open('xb') as stream:
            stream.write(b'ptb-cache')
        probe.unlink()
        if not marker.exists():
            atomic_json(marker, {'schema': SCHEMA})
    return path


def configuration_path():
    return user_data_root() / 'portable-cache.json'


def previous_directories():
    """Only bounded, explicitly recorded dedicated directories; no disk search."""
    try:
        config = _plain(configuration_path())
        if config.stat().st_size > 8192:
            return []
        value = json.loads(config.read_text('utf8'))
        paths = value.get('previousDirectories', []) if isinstance(value, dict) else []
        return [_validate_directory(path) for path in paths[:16] if isinstance(path, str)] if isinstance(paths, list) else []
    except (OSError, ValueError):
        return []


def read_directory():
    path = _plain(configuration_path())
    if not path.exists():
        return None
    if path.stat().st_size > 8192:
        raise ValueError('缓存位置记录无法读取，请重新选择目录。')
    value = json.loads(path.read_text('utf8'))
    if not isinstance(value, dict) or value.get('schema') != SCHEMA or not isinstance(value.get('directory'), str):
        raise ValueError('缓存位置记录无法读取，请重新选择目录。')
    return _validate_directory(value['directory'], create=True)


def configure_launcher(executable, arguments, choose):
    """Return False on cancellation; never substitute a hidden portable cache."""
    # The verified updater helper retains its parent's cache identity. An
    # ordinary external EXE always resolves its own package kind and preference.
    if arguments[:1] == ['--ptb-apply-update'] and os.environ.get(MODE_ENVIRONMENT) in ('portable', 'installer'):
        if os.environ.get(ENVIRONMENT):
            _validate_directory(os.environ[ENVIRONMENT], create=True)
        return True
    os.environ.pop(ENVIRONMENT, None)
    kind = distribution_kind(executable)
    os.environ[MODE_ENVIRONMENT] = kind
    if kind == 'installer':
        return True
    from .startup_cache import FileLock, atomic_json
    config = _plain(configuration_path())
    config.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(config.with_suffix('.lock')):
        previous = previous_directories()
        try:
            old = json.loads(config.read_text('utf8')).get('directory')
            if isinstance(old, str):
                previous.append(_validate_directory(old))
        except (OSError, ValueError, AttributeError):
            pass
        try:
            directory = read_directory()
            reason = ''
        except (OSError, ValueError) as error:
            directory, reason = None, str(error)
        while directory is None:
            parent = choose(reason)
            if parent is None:
                return False
            try:
                directory = _validate_directory(selected_directory(parent), create=True)
            except (OSError, ValueError) as error:
                reason = str(error)
        previous = list(dict.fromkeys(str(path) for path in previous if path != directory))[-16:]
        atomic_json(config, {'schema': SCHEMA, 'directory': str(directory), 'previousDirectories': previous})
        os.environ[ENVIRONMENT] = str(directory)
    return True


def portable_directory():
    value = os.environ.get(ENVIRONMENT)
    return _validate_directory(value) if value else None


def startup_directory():
    directory = portable_directory()
    return (directory if directory else user_data_root()) / 'startup-cache'


def browser_directory():
    directory = portable_directory()
    return directory / 'browser-cache' if directory else windows_workbench_storage() / 'http-cache'


def update_directory():
    directory = portable_directory()
    return (directory if directory else user_data_root()) / 'updates'


def location_status():
    directory = portable_directory()
    current = directory or startup_directory()
    previous = user_data_root() / 'startup-cache'
    return {'path': str(current), 'custom': directory is not None,
            'legacy_path': str(previous) if directory and previous.is_dir() else None}


def open_directory(*, legacy=False):
    """Only these native-owned paths can be opened, never a caller's path."""
    from PyQt6.QtCore import QUrl
    from PyQt6.QtGui import QDesktopServices
    status = location_status()
    value = status['legacy_path'] if legacy else status['path']
    if value is None:
        raise ValueError('旧缓存目录已不存在。')
    path = _plain(value)
    if not path.is_dir():
        raise ValueError('缓存目录已不存在，再次启动时会在所选位置重新准备。')
    if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
        raise OSError('缓存目录未能打开，请复制路径后在文件资源管理器中打开。')
    return {'opened': True}
