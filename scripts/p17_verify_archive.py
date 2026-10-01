"""Read back a P17 executable and compare its exact source/frontend payload."""
import argparse
import hashlib
import json
from pathlib import Path

from PyInstaller.archive.readers import CArchiveReader

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    if not args.name.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Plain artifact name required')
    work = ROOT / 'output' / ('build-' + args.name)
    artifact = ROOT / 'dist' / args.name / (args.name + '.exe')
    archive = CArchiveReader(str(artifact))
    names = {name.replace('\\', '/'): name for name in archive.toc}
    files = []
    for relative in ('backend/src', 'desktop/src', 'packages/phonetic_core/src', 'frontend/dist', 'docs/manual'):
        source = ROOT / relative if relative in ('frontend/dist', 'docs/manual') else work / 'snapshot' / relative
        for path in sorted(source.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            target = relative + '/' + path.relative_to(source).as_posix()
            assert target in names, 'Missing bundled file: ' + target
            sha = digest(path.read_bytes())
            assert digest(archive.extract(names[target])) == sha, 'Stale bundled file: ' + target
            current = ROOT / target
            assert current.is_file() and digest(current.read_bytes()) == sha, 'Checkout changed since build: ' + target
            files.append(dict(path=target, sha256=sha))
    config = json.loads(archive.extract(names['local-preview.json']))
    assert config['portable'] is False
    assert not any(name.startswith('phonetic_toolbox/') for name in names)
    report = dict(success=True, files_checked=len(files), archive_entries=len(names),
                  executable=artifact.name, bytes=artifact.stat().st_size,
                  sha256=digest(artifact.read_bytes()), configuration=config, files=files)
    (work / 'p17-archive-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in ('files', 'configuration')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
