"""Copy existing exact package notices without installing or altering runtimes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    sys.stdout.reconfigure(encoding='utf8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--host-site', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    conda = []
    for record in sorted((args.runtime / 'conda-meta').glob('*.json')):
        row = json.loads(record.read_text('utf8'))
        filename = row['fn']
        stem = filename.removesuffix('.conda').removesuffix('.tar.bz2')
        archive = args.cache / filename
        assert archive.is_file(), filename
        digest = sha(archive)
        assert digest == row['sha256'], filename
        directory = args.cache / stem / 'info/licenses'
        notices = []
        if directory.is_dir():
            for source in sorted(directory.rglob('*')):
                if source.is_file():
                    assert not source.is_symlink()
                    target = out / 'conda' / stem / source.relative_to(directory)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    notices.append(dict(path=target.relative_to(out).as_posix(), sha256=sha(target)))
        conda.append(dict(name=row['name'], version=row['version'], build=row['build'],
                          archive=filename, archiveSha256=digest, archiveUrl=row.get('url'),
                          license=row.get('license'), notices=notices))
    host = []
    for metadata in sorted(args.host_site.glob('*.dist-info')):
        notices = []
        for source in sorted(metadata.rglob('*')):
            if source.is_file() and source.name.lower().startswith(('licen', 'copying', 'notice', 'authors')):
                target = out / 'host-python' / metadata.name / source.relative_to(metadata)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                notices.append(dict(path=target.relative_to(out).as_posix(), sha256=sha(target)))
        if notices:
            host.append(dict(distribution=metadata.name, notices=notices))
    report = dict(schema='ptb-package-notices/1', conda=conda, host=host,
                  note='Notice files are preserved from exact local packages; this is not a corresponding-source archive or a final distribution-permission decision.')
    (out / 'inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', 'utf8')
    print(json.dumps(dict(condaPackages=len(conda), condaWithNotices=sum(bool(c['notices']) for c in conda),
                         hostDistributions=len(host)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
