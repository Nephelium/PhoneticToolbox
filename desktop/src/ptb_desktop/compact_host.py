"""First frozen hook: restore original host binaries before Qt startup hooks."""
import json
import hashlib
import os
from pathlib import Path
import re
import tarfile

from .compact_runtime import digest, windows_parts


def safe_target(root, name):
    target = root.joinpath(*windows_parts(name))
    if target.exists() or any(p.is_symlink() or (p.exists() and getattr(p.lstat(), 'st_file_attributes', 0) & 0x400)
                              for p in (target, *target.parents) if p.is_relative_to(root)):
        raise ValueError('Compact host target collides with the bootstrap')
    return target


def restore_shared(root, files, shared):
    stats = dict(linkedFiles=0, copiedFiles=0, linkedBytes=0, copiedBytes=0)
    checked = set()
    for name, source_name in shared.items():
        source = root.joinpath(*windows_parts(source_name))
        if not source.is_file() or any(p.is_symlink() or getattr(p.lstat(), 'st_file_attributes', 0) & 0x400
                                      for p in (source, *source.parents) if p.is_relative_to(root)):
            raise ValueError('Invalid shared dependency source')
        row = files[name]
        if source_name not in checked:
            if source.stat().st_size != row['size'] or digest(source) != row['sha256']:
                raise ValueError('Shared dependency source checksum mismatch')
            checked.add(source_name)
        target = safe_target(root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            # Exclusive creation never replaces an existing path. This also
            # supports volumes without hard links, preserving original bytes.
            checksum, received = hashlib.sha256(), 0
            with source.open('rb') as inp, target.open('xb') as output:
                while block := inp.read(256 * 1024):
                    received += len(block)
                    if received > row['size']:
                        raise ValueError('Shared copy exceeds budget')
                    checksum.update(block); output.write(block)
            if received != row['size'] or checksum.hexdigest() != row['sha256']:
                raise ValueError('Shared copy checksum mismatch')
            stats['copiedFiles'] += 1; stats['copiedBytes'] += received
        else:
            if not os.path.samefile(source, target):
                raise ValueError('Shared dependency link identity mismatch')
            stats['linkedFiles'] += 1; stats['linkedBytes'] += row['size']
    return stats


def expand(root):
    root = Path(root).resolve()
    descriptor = root / 'host-archive.json'
    if not descriptor.is_file():
        raise ValueError('Compact host archive descriptor is missing')
    payload = json.loads(descriptor.read_text('utf8'))
    marker = root / '.ptb-host-ready.json'
    expected_marker = {'schema': 'ptb-host-ready/1', 'sha256': payload.get('sha256')}
    version2 = payload.get('schema') == 'ptb-host-archive/2'
    if version2:
        expected_marker.update(schema='ptb-host-ready/2', manifestSha256=payload.get('manifestSha256'))
    if marker.is_file():
        completed = json.loads(marker.read_text('utf8'))
        if (completed == expected_marker if not version2 else
                {k:completed.get(k) for k in expected_marker} == expected_marker):
            return
        raise ValueError('Unexpected compact host completion marker')
    archive, manifest = root / 'host-payload.tar.xz', root / 'host-files.json'
    if (payload.get('schema') not in {'ptb-host-archive/1', 'ptb-host-archive/2'} or archive.stat().st_size != payload.get('size') or
            digest(archive) != payload.get('sha256') or manifest.stat().st_size > 12 * 1024**2 or
            digest(manifest) != payload.get('manifestSha256')):
        raise ValueError('Compact host payload checksum mismatch')
    value = json.loads(manifest.read_text('utf8'))
    files = value.get('files')
    if value.get('schema') != ('ptb-host-files/2' if version2 else 'ptb-host-files/1') or not isinstance(files, dict) or not 0 < len(files) <= 10000:
        raise ValueError('Invalid compact host file manifest')
    if sum(row['size'] for row in files.values()) != payload.get('expandedBytes') or payload['expandedBytes'] > 2 * 1024**3 or len(files) != payload.get('files'):
        raise ValueError('Compact host exceeds its expansion budget')
    names = set()
    for name, row in files.items():
        windows_parts(name)
        if name.casefold() in names or type(row.get('size')) is not int or not 0 <= row['size'] <= 512 * 1024**2 or not re.fullmatch('[0-9a-f]{64}',row.get('sha256','')):
            raise ValueError('Invalid compact host member')
        names.add(name.casefold())
    shared = value.get('sharedFiles', {}) if version2 else {}
    if not isinstance(shared, dict) or not set(shared).issubset(files):
        raise ValueError('Invalid shared dependency map')
    if version2:
        from .compact_runtime import expand as expand_science, relative_parts
        science_descriptor = json.loads((root/'desktop-bundle.json').read_text('utf8'))['runtimeArchive']
        if (science_descriptor['sha256'] != value.get('runtimeArchiveSha256') or
                science_descriptor['manifestSha256'] != value.get('runtimeManifestSha256') or
                digest(root/'runtime-files.json') != value['runtimeManifestSha256']):
            raise ValueError('Shared scientific identity mismatch')
        science = json.loads((root/'runtime-files.json').read_text('utf8'))['files']
        for name, source in shared.items():
            relative_parts(source)
            if science.get(source) != files[name]:
                raise ValueError('Shared dependency does not match original host bytes')
            safe_target(root, name)
        if (payload.get('sharedFiles') != len(shared) or
                payload.get('sharedBytes') != sum(files[name]['size'] for name in shared)):
            raise ValueError('Shared dependency budget mismatch')
        expand_science(root)
    expected_members = set(files) - set(shared)
    seen = set()
    with tarfile.open(archive, 'r:xz') as stream:
        for member in stream:
            if member.name in seen or member.name not in expected_members or not member.isfile():
                raise ValueError('Unexpected compact host archive member')
            row = files[member.name]
            if member.size != row['size']:
                raise ValueError('Compact host member size mismatch')
            target = safe_target(root, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            checksum, received = hashlib.sha256(), 0
            with stream.extractfile(member) as source, target.open('xb') as output:
                while block := source.read(256 * 1024):
                    received += len(block)
                    if received > member.size:
                        raise ValueError('Compact host member exceeds budget')
                    checksum.update(block); output.write(block)
            if received != row['size'] or checksum.hexdigest() != row['sha256']:
                raise ValueError('Compact host member checksum mismatch')
            seen.add(member.name)
    if seen != expected_members:
        raise ValueError('Incomplete compact host archive')
    if version2:
        expected_marker['restoration'] = restore_shared(root, files, shared)
    marker.write_text(json.dumps(expected_marker), 'utf8')
