"""Build an immutable complete portable archive and file inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def main():
    sys.stdout.reconfigure(encoding='utf8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    assert package.name == 'PhoneticToolbox'
    assert (package / 'PhoneticToolbox.exe').read_bytes()[:2] == b'MZ'
    assert (package / '_internal/desktop-bundle.json').is_file()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    version = json.loads((package / '_internal/desktop-bundle.json').read_text('utf8'))['version']
    archive = output / f'PhoneticToolbox-{version}-windows-x64-portable.zip'
    files = {}
    expanded = 0
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as writer:
        for file in sorted(package.rglob('*')):
            assert not file.is_symlink() and not getattr(file.lstat(), 'st_file_attributes', 0) & 0x400
            if not file.is_file():
                continue
            name = file.relative_to(package).as_posix()
            with file.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            files[name] = dict(size=file.stat().st_size, sha256=digest)
            expanded += file.stat().st_size
            writer.write(file, 'PhoneticToolbox/' + name)
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    with zipfile.ZipFile(archive) as reader:
        assert len(reader.infolist()) == len(files) <= 80000
        assert sum(f.file_size for f in reader.infolist()) == expanded <= 20 * 1024**3
        assert all(f.file_size <= 4*1024**3 and f.file_size <= max(1, f.compress_size)*1000 for f in reader.infolist())
    report = dict(schema='ptb-package-inventory/1', version=version, archive=str(archive),
                  size=archive.stat().st_size, sha256=digest, expandedBytes=expanded,
                  fileCount=len(files), files=files)
    (output / 'portable-inventory.json').write_text(json.dumps(report, indent=2)+'\n', 'utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}), flush=True)


if __name__ == '__main__':
    main()
