"""Preserve analyzed host native bytes, with stronger compression than PE zlib."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from datetime import datetime


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def shared_manifest(records, runtime_archive):
    """Match contents only; bind the mapping to the exact scientific archive."""
    if runtime_archive is None:
        return dict(schema='ptb-host-files/1', files=records)
    runtime_archive = Path(runtime_archive)
    descriptor = json.loads((runtime_archive.parent/'desktop-bundle.json').read_text('utf8'))['runtimeArchive']
    manifest = runtime_archive/'runtime-files.json'
    if (digest(manifest) != descriptor['manifestSha256'] or
            digest(runtime_archive/'runtime-payload.tar.xz') != descriptor['sha256']):
        raise ValueError('Scientific archive identity changed before host deduplication')
    science = json.loads(manifest.read_text('utf8'))
    if science.get('schema') != 'ptb-runtime-files/1':
        raise ValueError('Unknown scientific file manifest')
    by_hash = {}
    for name, row in sorted(science['files'].items()):
        by_hash.setdefault((row['size'], row['sha256']), name)
    shared = {name:by_hash[(row['size'], row['sha256'])] for name,row in records.items()
              if (row['size'], row['sha256']) in by_hash}
    return dict(schema='ptb-host-files/2', files=records, sharedFiles=shared,
                runtimeArchiveSha256=descriptor['sha256'], runtimeManifestSha256=descriptor['manifestSha256'])


def pack(analysis, output, runtime_archive=None, application_native=False):
    output = Path(output)
    reuse = output.exists()
    if not reuse:
        output.mkdir(parents=True, exist_ok=False)
    retained, compressed = [], []
    qt_bin = Path(next(iter(importlib.util.find_spec('PyQt6').submodule_search_locations))) / 'Qt6/bin'
    qt_identity = {}
    for entry in analysis.binaries:
        name, source, kind = entry
        path = Path(name.replace('\\', '/'))
        if path.name.startswith('Qt6') and path.suffix == '.dll':
            original = qt_bin / path.name
            if not original.is_file() or digest(source) != digest(original):
                raise RuntimeError('Host Qt DLL identity mismatch: ' + name)
            qt_identity[name] = digest(source)
        # Keep the CPython/stdlib extraction bootstrap and its own DLLs outside
        # the archive. Third-party modules and Qt are unpacked by the first hook.
        standard_extension = path.suffix == '.pyd' and Path(source).resolve().is_relative_to(Path(sys.base_prefix).resolve() / 'DLLs')
        bootstrap = (len(path.parts) == 1 and
                     (standard_extension or path.name.lower().startswith(('python', 'vcruntime', 'msvcp', 'libcrypto', 'libssl'))))
        # A small project/algorithm engine rebuild must not invalidate Qt.
        if application_native and path.as_posix().startswith('resources/'):
            retained.append(entry)
            continue
        if kind in ('BINARY', 'EXTENSION') and path.suffix.lower() in ('.dll', '.pyd', '.exe') and not bootstrap:
            compressed.append(entry)
        else:
            retained.append(entry)
    analysis.binaries = retained
    retained_data = []
    for entry in analysis.datas:
        name, source, kind = entry
        if name.replace('\\', '/').startswith('PyQt6/Qt6/') and kind == 'DATA':
            compressed.append(entry)
        else:
            retained_data.append(entry)
    if not qt_identity or not compressed:
        raise RuntimeError('Compact host did not preserve a reviewed Qt runtime')
    files = {}
    for name, source, kind in compressed:
        key = name.replace('\\', '/')
        if key in files:
            raise RuntimeError('Duplicate host archive target: ' + key)
        files[key] = dict(size=Path(source).stat().st_size, sha256=digest(source), source=source, kind=kind)
    archive = output / 'host-payload.tar.xz'
    manifest = output / 'host-files.json'
    records = {name: dict(size=row['size'], sha256=row['sha256']) for name, row in files.items()}
    contents = shared_manifest(records, runtime_archive)
    shared = contents.get('sharedFiles', {})
    schema = 'ptb-host-archive/2' if runtime_archive is not None else 'ptb-host-archive/1'
    descriptor = output / 'host-archive.json'
    if reuse:
        previous = json.loads(manifest.read_text('utf8'))
        metadata = json.loads(descriptor.read_text('utf8'))
        if (previous != contents or metadata.get('schema') != schema or archive.stat().st_size != metadata['size'] or
                digest(archive) != metadata['sha256'] or digest(manifest) != metadata['manifestSha256']):
            # Keep the rejected copy for diagnosis and compress the current inputs.
            # Do this in the same analysis pass, without accepting mismatched bytes.
            rejected = output.with_name(output.name + '-cache-rejected-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            output.rename(rejected)
            output.mkdir(parents=True, exist_ok=False)
            print('Host archive cache changed; rebuilding from current native inputs.', flush=True)
        else:
            analysis.datas = retained_data + [(path.name, str(path), 'DATA') for path in (archive, manifest, descriptor)]
            return
    with tarfile.open(archive, 'w:xz', preset=9) as stream:
        for name in sorted(files, key=lambda n: (files[n]['sha256'], n)):
            if name in shared:
                continue
            info = stream.gettarinfo(files[name]['source'], arcname=name)
            info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0
            with Path(files[name]['source']).open('rb') as content:
                stream.addfile(info, content)
    manifest.write_text(json.dumps(contents, indent=2) + '\n', 'utf8')
    descriptor.write_text(json.dumps(dict(schema=schema, size=archive.stat().st_size,
                         sha256=digest(archive), manifestSha256=digest(manifest),
                         expandedBytes=sum(row['size'] for row in files.values()), files=len(files),
                         sharedFiles=len(shared), sharedBytes=sum(records[n]['size'] for n in shared)), indent=2) + '\n', 'utf8')
    (output / 'host-qt-identity.json').write_text(json.dumps(qt_identity, indent=2) + '\n', 'utf8')
    (output / 'host-selection.json').write_text(json.dumps(files, indent=2) + '\n', 'utf8')
    analysis.datas = retained_data + [(path.name, str(path), 'DATA') for path in (archive, manifest, descriptor)]
