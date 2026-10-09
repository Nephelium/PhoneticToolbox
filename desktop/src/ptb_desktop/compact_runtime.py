"""Expand embedded scientific files inside the one-file bootloader's own folder."""
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time


def publish_directory(source, destination, *, timeout=3.0):
    """Retry transient Windows locks; never replace an existing destination."""
    deadline=time.monotonic()+timeout
    while True:
        if destination.exists():
            raise FileExistsError('Runtime destination already exists')
        try:
            source.rename(destination)
            return
        except OSError as error:
            # A real owned installation test observed WinError 5 at this exact
            # rename; the identical operation succeeded later. The lock owner
            # is unknown. Permanent permissions and all other errors propagate.
            if getattr(error,'winerror',None) not in (5,32,33) or time.monotonic()>=deadline:
                raise
            time.sleep(min(.1,max(0,deadline-time.monotonic())))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def windows_parts(name):
    if not isinstance(name, str) or len(name) > 1024 or '\\' in name or '\x00' in name:
        raise ValueError('Invalid runtime archive path')
    parts = name.split('/')
    if any(not p or p in ('.', '..') or len(p) > 255 or p.endswith((' ', '.')) or
           any(ord(c) < 32 or c in '<>:"|?*' for c in p) or
           re.fullmatch(r'(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', p, re.I) for p in parts):
        raise ValueError('Invalid Windows runtime path')
    return parts


def relative_parts(name):
    parts = windows_parts(name)
    if len(parts) < 3 or parts[:2] not in (['runtimes', 'egg'], ['runtimes', 'm05']):
        raise ValueError('Unexpected runtime archive root')
    return parts


def expand(root):
    root = Path(root).resolve()
    value = json.loads((root / 'desktop-bundle.json').read_text('utf8'))
    payload = value.get('runtimeArchive')
    if payload is None:
        return
    if (payload.get('schema') != 'ptb-runtime-archive/1' or
            payload.get('archive') != 'runtime-payload.tar.xz' or
            payload.get('manifest') != 'runtime-files.json'):
        raise ValueError('Invalid embedded runtime descriptor')
    marker = root / 'runtimes' / '.ptb-runtime-complete.json'
    if marker.is_file():
        completed = json.loads(marker.read_text('utf8'))
        if completed == {'schema': 'ptb-runtime-ready/1', 'archiveSha256': payload['sha256']}:
            return
        raise ValueError('Unexpected existing scientific runtime')
    archive, manifest = root / payload['archive'], root / payload['manifest']
    if archive.stat().st_size != payload.get('size') or digest(archive) != payload.get('sha256') or digest(manifest) != payload.get('manifestSha256'):
        raise ValueError('Embedded runtime payload checksum mismatch')
    if manifest.stat().st_size > 12 * 1024**2:
        raise ValueError('Embedded runtime manifest exceeds budget')
    records = json.loads(manifest.read_text('utf8'))
    files = records.get('files')
    if records.get('schema') != 'ptb-runtime-files/1' or not isinstance(files, dict) or not 0 < len(files) <= 20000:
        raise ValueError('Invalid embedded runtime file manifest')
    case_paths = set()
    for name, row in files.items():
        relative_parts(name)
        if name.casefold() in case_paths or type(row.get('size')) is not int or not 0 <= row['size'] <= 512 * 1024**2 or not re.fullmatch('[0-9a-f]{64}', row.get('sha256', '')):
            raise ValueError('Invalid embedded runtime member')
        case_paths.add(name.casefold())
    total = sum(row['size'] for row in files.values())
    if total > 2 * 1024**3 or total != payload.get('expandedBytes') or len(files) != payload.get('files'):
        raise ValueError('Embedded runtime expansion exceeds budget')
    target = root / '.ptb-runtime-expanding'
    target.mkdir(exist_ok=False)
    seen = set()
    with tarfile.open(archive, 'r:xz') as stream:
        for member in stream:
            parts = relative_parts(member.name)
            if not member.isfile() or member.name not in files or member.name in seen:
                raise ValueError('Unexpected embedded runtime archive member')
            row = files[member.name]
            if member.size != row['size']:
                raise ValueError('Embedded runtime member size mismatch')
            seen.add(member.name)
            path = target.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            checksum, received = hashlib.sha256(), 0
            with stream.extractfile(member) as source, path.open('xb') as output:
                while block := source.read(256 * 1024):
                    received += len(block)
                    if received > member.size:
                        raise ValueError('Embedded runtime member exceeds budget')
                    checksum.update(block)
                    output.write(block)
            if received != row['size'] or checksum.hexdigest() != row['sha256']:
                raise ValueError('Embedded runtime member checksum mismatch')
    if seen != set(files):
        raise ValueError('Embedded runtime archive is incomplete')
    publish_directory(target / 'runtimes',root / 'runtimes')
    marker.write_text(json.dumps({'schema': 'ptb-runtime-ready/1', 'archiveSha256': payload['sha256']}), 'utf8')
