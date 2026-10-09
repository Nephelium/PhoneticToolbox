"""Build an opaque, byte-preserving archive for the two scientific processes."""
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(stage, destination, cache=None):
    stage, destination = Path(stage), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    sources = {}
    for runtime in ('egg', 'm05'):
        for path in sorted((stage / runtime).rglob('*')):
            if path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
                raise ValueError('Runtime stage contains a link')
            if not path.is_file():
                continue
            relative = 'runtimes/' + path.relative_to(stage).as_posix()
            files[relative] = dict(size=path.stat().st_size, sha256=digest(path))
            sources[relative] = path
    if not files or len(files) > 20000 or sum(row['size'] for row in files.values()) > 2 * 1024**3:
        raise ValueError('Scientific runtime payload exceeds its reviewed budget')
    manifest = destination / 'runtime-files.json'
    manifest.write_text(json.dumps(dict(schema='ptb-runtime-files/1', files=files), indent=2) + '\n', 'utf8')
    archive = destination / 'runtime-payload.tar.xz'
    if cache is not None:
        import shutil
        cache = Path(cache)
        descriptor = json.loads((cache.parent / 'desktop-bundle.json').read_text('utf8'))['runtimeArchive']
        previous = json.loads((cache / 'runtime-files.json').read_text('utf8'))
        if previous != dict(schema='ptb-runtime-files/1', files=files) or digest(cache / archive.name) != descriptor['sha256']:
            raise ValueError('Cached runtime archive differs from the selected stage')
        shutil.copyfile(cache / archive.name, archive)
        return descriptor
    # Identical files from the isolated environments are neighbours in the solid
    # stream. Preserve both paths and every original byte; do not merge versions.
    with tarfile.open(archive, 'w:xz', preset=9) as output:
        for relative in sorted(files, key=lambda p: (files[p]['sha256'], p)):
            info = output.gettarinfo(str(sources[relative]), arcname=relative)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 0
            with sources[relative].open('rb') as stream:
                output.addfile(info, stream)
    return dict(schema='ptb-runtime-archive/1', archive=archive.name,
                size=archive.stat().st_size, sha256=digest(archive),
                manifest=manifest.name, manifestSha256=digest(manifest),
                expandedBytes=sum(row['size'] for row in files.values()), files=len(files))
