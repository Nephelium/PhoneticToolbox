"""Restore verified scientific bytes to disposable scratch, then compare a new subset.

Never edits a development environment or the source distribution. The selected
stage is a build input; generated probe media and restored originals are scratch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
from types import SimpleNamespace

from source_snapshot import freeze
from verification_artifacts import verification_run, write_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'release'))
from prepare_lean_runtimes import prepare, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True, help='Previously verified compact build snapshot')
    parser.add_argument('--stage', type=Path, required=True, help='New output/release-staging directory')
    args = parser.parse_args()
    stage = args.stage.resolve()
    if not stage.is_relative_to(ROOT / 'output/release-staging') or stage.exists():
        parser.error('Choose a new owned output/release-staging directory')
    cache = args.cache.resolve()
    if not cache.is_relative_to(ROOT / 'output'):
        parser.error('Expected an owned build snapshot')
    bundle = json.loads((cache / 'desktop-bundle.json').read_text('utf8'))
    if set(bundle['runtimes']) != {'PTB_EGG_PYTHON', 'PTB_M05_PYTHON'} or bundle.get('mfa'):
        raise ValueError('Expected a previously inventoried non-MFA compact bundle')
    descriptor = bundle['runtimeArchive']
    archive = cache / 'runtime-archive/runtime-payload.tar.xz'
    manifest = cache / 'runtime-archive/runtime-files.json'
    if sha(archive) != descriptor['sha256'] or sha(manifest) != descriptor['manifestSha256']:
        raise ValueError('Source scientific archive identity differs')
    records = json.loads(manifest.read_text('utf8'))['files']
    with verification_run('package-trim', 'runtime', dict(command=sys.argv, sourceArchive=descriptor)) as (out, scratch):
        print(json.dumps({'evidence': str(out), 'stage': str(args.stage.resolve())}), flush=True)
        original = scratch / 'original/_internal'
        seen = set()
        with tarfile.open(archive, 'r|xz') as stream:
            for member in stream:
                path = PurePosixPath(member.name)
                if (not member.isfile() or path.is_absolute() or '..' in path.parts or ':' in member.name
                        or member.name not in records or member.name in seen):
                    raise ValueError('Unexpected runtime archive member: ' + member.name)
                expected = records[member.name]
                if member.size != expected['size']:
                    raise ValueError('Runtime size differs: ' + member.name)
                dest = original.joinpath(*path.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as src, dest.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                if sha(dest) != expected['sha256']:
                    raise ValueError('Runtime bytes differ: ' + member.name)
                seen.add(member.name)
        if seen != records.keys():
            raise ValueError('Scientific archive is incomplete')
        # Both sides run the same immutable current scientific source.
        freeze(ROOT, original)
        shutil.copytree(ROOT / 'resources/m05', original / 'resources/m05')
        for source, target in (
                ('phonetic_toolbox/core/acoustic/reaper.exe', 'resources/research/reaper.exe'),
                ('tests/fixtures/m03/EGG-SYN-PCM16.npz', 'preview-fixtures/EGG-SYN-PCM16.npz')):
            dest = original / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, dest)
        validation = scratch / 'probe'
        try:
            code = prepare(SimpleNamespace(source_bundle=original.parent, stage=args.stage, validation_root=validation))
        finally:
            for source in validation.rglob('*'):
                if source.is_file() and source.suffix in {'.json', '.log'}:
                    dest = out / 'probe' / source.relative_to(validation)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, dest)
        write_json(out / 'stage-identity.json', dict(stage=str(args.stage.resolve()),
                   selectionSha256=sha(args.stage / 'selection-manifest.json'),
                   sourceSnapshotSha256=sha(original / 'source-snapshot.json'), returncode=code))
        print(json.dumps({'returncode': code, 'evidence': str(out)}), flush=True)
        return code


if __name__ == '__main__':
    raise SystemExit(main())
