"""Single-file MFA delivery. Trust is pinned in the app, never supplied by a ZIP."""
import hashlib
import json
from pathlib import Path
import stat
from threading import Lock
import time
import zipfile

from . import release_catalog
from .components import ComponentManager, digest, no_links, safe_name

MANIFEST = 'component-manifest.json'
_lock = Lock()
_state = {'stage': 'idle', 'completed': 0, 'total': 0}


def status():
    value = dict(_state)
    if time.monotonic() - value.pop('sample_time', 0) > 3:
        value['bytes_per_second'] = 0
    return value


def progress(stage, completed=0, total=0):
    global _state
    _state = dict(stage=stage, completed=completed, total=total, bytes_per_second=0, sample_time=time.monotonic())


def download_progress():
    """Rate uses bytes transferred this session; resumed bytes are not counted."""
    last_time, last_bytes, speed = time.monotonic(), None, 0
    def update(done, total):
        nonlocal last_time, last_bytes, speed
        global _state
        now = time.monotonic()
        if last_bytes is None:
            last_time, last_bytes = now, done
        elif now - last_time >= .5:
            speed = max(0, (done - last_bytes) / (now - last_time))
            last_time, last_bytes = now, done
        _state = dict(stage='downloading', completed=done, total=total,
                      bytes_per_second=speed, sample_time=now)
    return update


def release():
    value = release_catalog.RELEASE
    if not value or not value.get('sha256'):
        raise ValueError('m11_download_not_published')
    return dict(value)


def read_manifest(raw, pinned):
    if len(raw) > 32_000_000 or hashlib.sha256(raw).hexdigest() != pinned['manifest_sha256']:
        raise ValueError('m11_manifest_hash_mismatch')
    value = json.loads(raw)
    if value.get('schema') != 'm11-component/1' or value.get('id') != pinned['id']:
        raise ValueError('m11_manifest_invalid')
    # The archive hash cannot be embedded in the archive itself. Its identity is
    # pinned by the application; the internal manifest describes payload files.
    value.update(sha256=pinned['sha256'], download_bytes=pinned['download_bytes'])
    value['files'].append(dict(path=MANIFEST, bytes=len(raw), sha256=pinned['manifest_sha256']))
    value['installed_bytes'] += len(raw)
    return value


def archive_manifest(path, pinned):
    path = Path(path)
    no_links(path)
    progress('verifying')
    if path.stat().st_size != pinned['download_bytes'] or digest(path) != pinned['sha256']:
        raise ValueError('m11_archive_hash_mismatch')
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if m.filename == MANIFEST]
        if len(members) != 1 or members[0].file_size > 32_000_000:
            raise ValueError('m11_manifest_invalid')
        return read_manifest(archive.read(members[0]), pinned)


def folder_manifest(folder, pinned):
    folder = Path(folder).absolute()
    no_links(folder)
    path = folder / MANIFEST
    no_links(path)
    if not path.is_file():
        raise ValueError('m11_delivery_folder_required')
    if path.stat().st_size > 32_000_000:
        raise ValueError('m11_manifest_invalid')
    trusted = read_manifest(path.read_bytes(), pinned)
    expected = {str(safe_name(row['path'])): row for row in trusted['files']}
    if len(expected) != len(trusted['files']):
        raise ValueError('m11_manifest_invalid')
    # Check the entire directory before executing its Python. Extra .pth, DLL,
    # sitecustomize or bytecode files must not be silently trusted.
    found = set()
    for item in folder.rglob('*'):
        no_links(item)
        mode = item.stat().st_mode
        if stat.S_ISDIR(mode):
            continue
        name = item.relative_to(folder).as_posix()
        if name == 'ptb-component.json':
            continue  # host receipt, never imported or executed
        row = expected.get(name)
        if not stat.S_ISREG(mode) or row is None:
            raise ValueError('m11_folder_files_changed')
        if item.stat().st_size != row['bytes'] or digest(item) != row['sha256']:
            raise ValueError('m11_file_hash_mismatch')
        found.add(name)
        progress('verifying', len(found), len(expected))
    if found != set(expected):
        raise ValueError('m11_folder_files_changed')
    return trusted


def install(action, *, archive=None, folder=None, root=None):
    from .runtime import registry_root
    from .probe import register, publish_registration
    if not _lock.acquire(blocking=False):
        raise ValueError('m11_component_busy')
    try:
        pinned = release()
        manager = ComponentManager(root or registry_root())
        prepared = {}

        def check(target):
            progress('checking')
            prepared.update(register(target / 'env', target / 'models/mandarin_mfa.zip',
                                     target / 'models/mandarin_mfa.dict', root=manager.root, publish=False))
            return dict(success=True, receipt=prepared['receipt'])

        if action == 'install_folder':
            target = Path(folder).absolute()
            trusted = folder_manifest(target, pinned)
            check(target)
        else:
            if action == 'download':
                progress('downloading', 0, pinned['download_bytes'])
                archive = manager.download(pinned, progress=download_progress())
            trusted = archive_manifest(archive, pinned)
            manager.import_archive(archive, trusted, check,
                                   progress=lambda done, total: progress('extracting', done, total))
        prepared['runtime_record'].update(download_bytes=pinned['download_bytes'],
            source='PhoneticToolbox MFA 3.3.8-r1 (Windows x64)', archive_sha256=pinned['sha256'])
        publish_registration(prepared, manager.root)
        progress('complete')
        return dict(success=True, runtime_id=prepared['runtime_id'], model_id=prepared['model_id'])
    except Exception:
        progress('failed')
        raise
    finally:
        _lock.release()
