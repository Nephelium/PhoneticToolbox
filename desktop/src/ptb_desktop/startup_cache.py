"""Content-addressed Windows startup files; no Qt or scientific imports.

The user-facing launcher owns preparation and lifetime leases. This module is
also frozen into the application so cleanup uses the same ownership rules.
"""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import tarfile
import time
import uuid

from .compact_runtime import digest, publish_directory, windows_parts

SCHEMA = 'ptb-startup-cache/1'
COOKIE = struct.Struct('!8sIIII64s')
HEX = re.compile('[0-9a-f]{64}')


def cache_root():
    # Owned QA redirects LOCALAPPDATA, just as normal application data does.
    from .platform_paths import user_data_root
    return user_data_root() / 'startup-cache'


def plain(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    if not path.is_relative_to(root):
        raise ValueError('Cache path escapes its root')
    for item in (path, *path.parents):
        try: info = item.lstat()
        except FileNotFoundError: continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Cache paths must not traverse reparse points')
    return path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf8') as out:
        json.dump(value, out, ensure_ascii=False, separators=(',', ':'))
        out.flush(); os.fsync(out.fileno())
    os.replace(temporary, path)


class FileLock:
    def __init__(self, path):
        self.path, self.file = Path(path), None

    def acquire(self, timeout=120, notify=lambda: None):
        import msvcrt
        plain(self.path, self.path.parent)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.file.write(b'0'); self.file.flush()
        until = time.monotonic() + timeout
        while True:
            try:
                self.file.seek(0); msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                return self
            except OSError:
                if time.monotonic() >= until:
                    self.file.close(); self.file = None
                    raise TimeoutError('缓存正在被其他窗口准备或使用，请稍后重试。') from None
                notify(); time.sleep(.1)

    def close(self):
        if self.file:
            import msvcrt
            self.file.seek(0); msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            self.file.close(); self.file = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.close()


@contextmanager
def maintenance(root, notify=lambda: None):
    root = plain(root, Path(root).parent)
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(root.parent / '.startup-cache.lock').acquire(notify=notify)
    try:
        if root.exists():
            marker = plain(root / 'owner.json', root)
            if not marker.is_file() or json.loads(marker.read_text('utf8')) != {'schema': SCHEMA}:
                raise ValueError('Existing cache ownership could not be verified')
        else:
            root.mkdir()
            atomic_json(root / 'owner.json', {'schema': SCHEMA})
        yield root
    finally:
        lock.close()


def lease(root):
    path = plain(root / 'leases' / (uuid.uuid4().hex + '.lock'), root)
    return FileLock(path).acquire()


def active_leases(root):
    folder = plain(root / 'leases', root)
    active = 0
    if folder.is_dir():
        for path in folder.iterdir():
            if not re.fullmatch('[0-9a-f]{32}\\.lock', path.name):
                raise ValueError('Unexpected cache lease')
            lock = FileLock(path)
            try: lock.acquire(timeout=0)
            except TimeoutError: active += 1
            else:
                lock.close(); path.unlink()
    return active


def remove_owned(path, root):
    """Validate the final absolute tree before any recursive removal."""
    path = plain(path, root)
    if path == root or not path.is_dir():
        raise ValueError('Invalid cache removal target')
    for parent, dirs, files in os.walk(path, followlinks=False):
        for name in (*dirs, *files): plain(Path(parent) / name, root)
    shutil.rmtree(path)


def request_clear(root=None):
    root = cache_root() if root is None else Path(root)
    with maintenance(root):
        atomic_json(root / 'clear-request.json', {'schema': SCHEMA, 'requested': time.time()})
    return {'scheduled': True}


def clear_if_idle(root=None, *, requested_only=False, extra_cleanup=None):
    root = cache_root() if root is None else Path(root)
    if not root.exists(): return {'complete': True, 'active': 0}
    with maintenance(root):
        if requested_only and not (root / 'clear-request.json').is_file():
            return {'complete': True, 'active': 0, 'skipped': True}
        active = active_leases(root)
        if active: return {'complete': False, 'active': active}
        extra_ok = extra_cleanup().get('complete',False) if extra_cleanup else True
        for family in ('apps', 'host', 'science', 'staging', 'leases'):
            path = plain(root / family, root)
            if path.exists(): remove_owned(path, root)
        for name in ('clear-request.json', 'last-launch.json'):
            path = plain(root / name, root)
            if path.exists(): path.unlink()
        remaining = {path.name for path in root.iterdir()}
        if remaining == {'owner.json'}:
            (root / 'owner.json').unlink(); root.rmdir()
    return {'complete': extra_ok, 'active': 0}


class Segment(io.RawIOBase):
    def __init__(self, executable, offset, size):
        self.file = Path(executable).open('rb'); self.file.seek(offset)
        self.remaining = size

    def readable(self): return True

    def read(self, size=-1):
        size = self.remaining if size < 0 else min(size, self.remaining)
        data = self.file.read(size); self.remaining -= len(data)
        return data

    def close(self):
        self.file.close(); super().close()


class Payload:
    def __init__(self, executable, config):
        self.executable, self.config = Path(executable), config
        if config.get('schema') != 'ptb-cache-payload/1':
            raise ValueError('Unknown persistent payload format')
        with self.executable.open('rb') as file:
            file.seek(-COOKIE.size, 2)
            magic, package_size, *_ = COOKIE.unpack(file.read(COOKIE.size))
        if magic != b'MEI\014\013\012\013\016':
            raise ValueError('Invalid launcher archive footer')
        self.start = self.executable.stat().st_size - package_size - sum(row['size'] for row in config['blobs'].values())
        if self.start < 1024: raise ValueError('Invalid launcher payload offset')
        self.offsets = {}; offset = self.start
        for name, row in config['blobs'].items():
            if type(row['size']) is not int or not 0 < row['size'] <= 500_000_000 or not HEX.fullmatch(row['sha256']):
                raise ValueError('Invalid embedded payload budget')
            self.offsets[name] = offset; offset += row['size']

    def open(self, name):
        return Segment(self.executable, self.offsets[name], self.config['blobs'][name]['size'])

    def check(self, name):
        checksum = hashlib.sha256()
        with self.open(name) as file:
            while block := file.read(1024 * 1024): checksum.update(block)
        if checksum.hexdigest() != self.config['blobs'][name]['sha256']:
            raise ValueError('程序文件校验失败，请重新下载完整安装包。')


def validate_records(records):
    names = set()
    if not isinstance(records, dict) or not 0 < len(records) <= 30000:
        raise ValueError('Invalid cache content manifest')
    for name, row in records.items():
        windows_parts(name)
        if name.casefold() in names or type(row.get('size')) is not int or not 0 <= row['size'] <= 512 * 1024**2 or not HEX.fullmatch(row.get('sha256', '')):
            raise ValueError('Invalid cache member')
        names.add(name.casefold())
    if sum(row['size'] for row in records.values()) > 3 * 1024**3:
        raise ValueError('Cache expansion exceeds budget')


class TreeGuard:
    """Check each ancestor once per locked operation and every leaf afresh.

    All actors in this application hold the cache maintenance lock. Ancestors
    of its private user root are validated at entry. Never retain this memo
    across launches or across release of the maintenance lock.
    """
    def __init__(self, root):
        self.root=plain(root,root)
        self.checked={self.root}

    def path(self,name):
        parts=windows_parts(name);path=self.root
        for part in parts[:-1]:
            path=path/part
            if path not in self.checked:
                try:info=path.lstat()
                except FileNotFoundError:continue
                if not stat.S_ISDIR(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
                    raise ValueError('Invalid cache directory')
                self.checked.add(path)
        target=path/parts[-1]
        try:info=target.lstat()
        except FileNotFoundError:return target,None
        if not stat.S_ISREG(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
            raise ValueError('Invalid cache file')
        return target,info


def verify_tree(folder, records, memo, progress=lambda *args: None):
    """Check every leaf and every unique file's bytes during this locked launch.

    A small pool overlaps file-open latency without retaining hashes across
    launches. Join all readers before a failed tree can be repaired or removed.
    """
    guard=TreeGuard(folder); pending={}; expected=[]; counts={}
    total=len(records); complete=0; previous=time.monotonic()
    progress(0,total)
    for name, row in records.items():
        path,info=guard.path(name)
        if info is None:return False
        if info.st_size != row['size']: return False
        key = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        actual = memo.get(key)
        expected.append((key,row['sha256']))
        if actual is not None:
            if actual != row['sha256']:return False
            complete+=1
        else:
            pending.setdefault(key,path)
            counts[key]=counts.get(key,0)+1
    progress(complete,total)
    if pending:
        with ThreadPoolExecutor(max_workers=4,thread_name_prefix='ptb-cache-check') as readers:
            for key,actual in zip(pending,readers.map(digest,pending.values())):
                memo[key]=actual;complete+=counts[key]
                now=time.monotonic()
                if now-previous>=.05 or complete==total:
                    progress(complete,total);previous=now
    if any(memo[key]!=checksum for key,checksum in expected):return False
    progress(total,total)
    return True


def extract(payload, blob, folder, records, progress):
    payload.check(blob)
    total = sum(row['size'] for row in records.values()); complete = 0; seen = set();guard=TreeGuard(folder)
    with payload.open(blob) as source, tarfile.open(fileobj=source, mode='r|xz') as archive:
        for member in archive:
            if not member.isfile() or member.name in seen or member.name not in records:
                raise ValueError('Unexpected cached archive member')
            row = records[member.name]
            if member.size != row['size']: raise ValueError('Cached member exceeds its budget')
            target,info=guard.path(member.name)
            if info is not None:raise ValueError('Cached archive member collides with existing file')
            target.parent.mkdir(parents=True, exist_ok=True)
            checksum = hashlib.sha256(); count = 0
            with archive.extractfile(member) as inp, target.open('xb') as out:
                while block := inp.read(256 * 1024):
                    count += len(block)
                    if count > row['size']: raise ValueError('Cached member expansion exceeds budget')
                    checksum.update(block); out.write(block)
                    complete += len(block)
                    progress(blob, complete, total)
            if count != row['size'] or checksum.hexdigest() != row['sha256']:
                raise ValueError('Cached member checksum mismatch')
            seen.add(member.name)
    if seen != set(records): raise ValueError('Incomplete cache payload')


def link_files(source, target, records, progress=lambda *args: None):
    linked = copied = 0
    source_guard,target_guard=TreeGuard(source),TreeGuard(target)
    for name, row in records.items():
        src,source_info=source_guard.path(name)
        dst,target_info=target_guard.path(name)
        if source_info is None:raise ValueError('Cached dependency source disappeared')
        dst.parent.mkdir(parents=True, exist_ok=True)
        if target_info is not None: raise ValueError('Cached dependency collides with application')
        try: os.link(src, dst); linked += 1
        except OSError:
            with src.open('rb') as inp, dst.open('xb') as out: shutil.copyfileobj(inp, out, 256 * 1024)
            if dst.stat().st_size != row['size'] or digest(dst) != row['sha256']:
                raise ValueError('Cached dependency copy mismatch')
            copied += 1
    return {'linked': linked, 'copied': copied}


def prepare(payload, root=None, progress=lambda *args: None):
    root = cache_root() if root is None else Path(root)
    config = payload.config
    started = time.monotonic()
    progress('checking',0,0)
    for component in config['components'].values(): validate_records(component['files'])
    memo = {}; report = {'prepared': [], 'reused': [], 'componentSeconds': {}}
    with maintenance(root, lambda: progress('waiting', 0, 0)):
        if (root / 'clear-request.json').exists():
            raise RuntimeError('已预约清除缓存，请先关闭其他 PhoneticToolbox 窗口，清理完成后再启动。')
        cache = {}
        staging = plain(root / 'staging', root)
        if staging.is_dir() and not active_leases(root):
            for previous in staging.iterdir():
                if not re.fullmatch('[0-9a-f]{32}', previous.name):
                    raise ValueError('Unknown interrupted cache directory')
                remove_owned(previous, root)
        for family in ('science', 'host', 'apps'):
            component_started=time.monotonic()
            item = config['components'][family]; identity = item['id']
            if not HEX.fullmatch(identity): raise ValueError('Invalid cache identity')
            # Keep native DLL paths below legacy Windows loader limits. The
            # full SHA remains authoritative in the marker, never the prefix.
            target = plain(root / family / identity[:32], root); content = target
            marker = target / 'ready.json'
            try:
                existing = json.loads(marker.read_text('utf8')) if marker.is_file() else {}
                if HEX.fullmatch(str(existing.get('identity',''))) and existing['identity'] != identity:
                    raise RuntimeError('缓存目录标识冲突，原文件已保留。')
                valid = existing == {'schema': SCHEMA, 'identity': identity}
            except (ValueError, OSError): valid = False
            def checked(done,total):progress('verify-'+family,done,total)
            if valid and verify_tree(content, item['files'], memo,checked):
                report['reused'].append(family); cache[family] = content
                report['componentSeconds'][family]=time.monotonic()-component_started
                continue
            progress('repair' if target.exists() else family, 0, 0)
            if target.exists():
                if active_leases(root):
                    raise RuntimeError('缓存校验未通过，请关闭其他窗口后重试，以便安全修复。')
                remove_owned(target, root)
            stage = plain(root / 'staging' / uuid.uuid4().hex, root)
            stage.mkdir(parents=True); folder = stage
            # This app is temporary during preparation; incomplete output can
            # never acquire a ready marker or be launched after interruption.
            if shutil.disk_usage(root).free < sum(r['size'] for r in item['files'].values()) + 64 * 1024**2:
                raise OSError('磁盘空间不足，请释放空间后重试。')
            if family == 'host':
                shared = config['sharedFiles']
                extract(payload, family, folder, {n:r for n,r in item['files'].items() if n not in shared}, progress)
                for name, source_name in shared.items():
                    src = cache['science'].joinpath(*windows_parts(source_name))
                    row = item['files'][name]
                    if config['components']['science']['files'].get(source_name) != row:
                        raise ValueError('Shared scientific content mismatch')
                    dst = plain(folder.joinpath(*windows_parts(name)), folder); dst.parent.mkdir(parents=True, exist_ok=True)
                    try: os.link(src, dst)
                    except OSError:
                        with src.open('rb') as inp, dst.open('xb') as out: shutil.copyfileobj(inp, out)
            elif family == 'science': extract(payload, family, folder, item['files'], progress)
            else:
                extract(payload, family, folder, config['applicationFiles'], progress)
                for dependencies in ('science', 'host'):
                    report[dependencies + 'Attachment'] = link_files(cache[dependencies], folder / '_internal', config['components'][dependencies]['files'])
            if not verify_tree(folder, item['files'], memo,checked): raise ValueError('Prepared cache failed content verification')
            atomic_json(stage / 'ready.json', {'schema': SCHEMA, 'identity': identity})
            target.parent.mkdir(parents=True, exist_ok=True)
            publish_directory(stage, target)
            report['prepared'].append(family); cache[family] = target
            report['componentSeconds'][family]=time.monotonic()-component_started
        live = lease(root)
        report.update(seconds=time.monotonic() - started, application=str(cache['apps'] / 'PhoneticToolbox.exe'))
        atomic_json(root / 'last-launch.json', report)
        return Path(report['application']), live, report


def external_executable():
    import sys
    # Only the frozen cache launcher supplies this path; scientific worker
    # dispatch still uses sys.executable, which is the inner application.
    value = os.environ.get('PTB_DISTRIBUTION_EXE') if getattr(sys, 'frozen', False) else None
    return Path(value or sys.executable).absolute()
