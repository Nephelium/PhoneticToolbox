"""Verify a completed one-file build, including both opaque embedded archives."""
import argparse
import hashlib
import json
from pathlib import Path

from read_distribution import DistributionReader

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--host-archive', default='host-archive')
    options = parser.parse_args()
    folder, work = options.package.resolve(), options.work.resolve()
    if not folder.is_relative_to(ROOT / 'dist') or not work.is_relative_to(ROOT / 'output'):
        raise ValueError('Expected an owned compact build')
    executable = folder / 'PhoneticToolbox.exe'
    if not 0 < executable.stat().st_size <= 500_000_000:
        raise ValueError('EXE exceeds the authorized limit')
    archive = DistributionReader(executable)
    if archive.cache:
        for name in archive.cache['blobs']:archive.payload.check(name)
        from ptb_desktop.startup_cache import validate_records
        from cached_launcher import json_hash
        for value in archive.cache['components'].values():
            validate_records(value['files'])
            if json_hash(value['files'])!=value['id']:raise ValueError('Persistent component identity mismatch')
    value = json.loads(archive.extract('desktop-bundle.json'))
    if value.get('mfa') is not None or set(value['runtimes']) != {'PTB_EGG_PYTHON', 'PTB_M05_PYTHON'}:
        raise ValueError('Compact runtime boundary differs from the requirement')
    identities = {}
    for prefix, manifest_name, descriptor in (
        ('runtime', 'runtime-files.json', value['runtimeArchive']),
        ('host', 'host-files.json', json.loads(archive.extract('host-archive.json')))):
        raw = archive.extract(prefix + '-payload.tar.xz')
        manifest = archive.extract(manifest_name)
        if len(raw) != descriptor['size'] or hashlib.sha256(raw).hexdigest() != descriptor['sha256'] or hashlib.sha256(manifest).hexdigest() != descriptor['manifestSha256']:
            raise ValueError('Embedded archive is not the verified build input')
        records = json.loads(manifest)['files']
        if len(records) != descriptor['files'] or sum(row['size'] for row in records.values()) != descriptor['expandedBytes']:
            raise ValueError('Embedded expansion budget mismatch')
        identities[prefix] = descriptor
    host_archive = (work / options.host_archive).resolve()
    if not host_archive.is_relative_to(work):
        raise ValueError('Host audit path escapes the owned build')
    qt = json.loads((host_archive / 'host-qt-identity.json').read_text('utf8'))
    host_files = json.loads(archive.extract('host-files.json'))['files']
    host_value = json.loads(archive.extract('host-files.json'))
    shared = host_value.get('sharedFiles', {})
    if host_value.get('schema') == 'ptb-host-files/2':
        science = json.loads(archive.extract('runtime-files.json'))['files']
        if (host_value['runtimeArchiveSha256'] != value['runtimeArchive']['sha256'] or
                host_value['runtimeManifestSha256'] != value['runtimeArchive']['manifestSha256'] or
                any(science.get(source) != host_files.get(target) for target,source in shared.items()) or
                identities['host'].get('sharedFiles') != len(shared)):
            raise ValueError('Shared native identity mismatch')
    if not qt or any(host_files[name]['sha256'] != sha for name, sha in qt.items()):
        raise ValueError('Embedded Qt identity differs from the reviewed host')
    with executable.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    application = dict(schema='ptb-desktop-release/1', version=value['version'], entry=executable.name,
                       layout='onefile/1', executable=dict(size=executable.stat().st_size, sha256=sha))
    previous = json.loads((folder/'build-info.json').read_text('utf8'))
    for name, metadata in [('application.json', application), ('build-info.json', previous | dict(
            kind='distributable-preview', portable=True, bundle_profile='compact-onefile-no-mfa/1',
            version=value['version'], exe=executable.name, bytes=executable.stat().st_size, sha256=sha,
            mfa=dict(bundled=False, environment='user-configured external environment', models='not_bundled'),
            runtimes=value['runtimes'], archives=identities, qtIdentity=qt,
            sharedFiles=len(shared), sharedBytes=sum(host_files[n]['size'] for n in shared)))]:
        (folder / name).write_text(json.dumps(metadata, indent=2) + '\n', 'utf8')
    print(json.dumps(dict(success=True, bytes=executable.stat().st_size, sha256=sha, hostQtDlls=len(qt))), flush=True)


if __name__ == '__main__':
    main()
