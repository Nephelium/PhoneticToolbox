"""One immutable set of project code for PyInstaller analysis and bundled workers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

CODE_ROOTS = ('backend/src', 'desktop/src', 'packages/phonetic_core/src')
IMPORT_ROOTS = ('scripts', *CODE_ROOTS)


def identity(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'size': path.stat().st_size, 'sha256': digest}


def selected_files(root):
    root = Path(root)
    files = list((root / 'scripts').glob('*.py'))
    for relative in CODE_ROOTS:
        files.extend(path for path in (root / relative).rglob('*') if path.is_file()
                     and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo'})
    return sorted(path.relative_to(root).as_posix() for path in files)


def freeze(root, snapshot):
    """Fail if files change during copying, including new or removed files."""
    root, snapshot = Path(root).resolve(), Path(snapshot).resolve()
    files = selected_files(root)
    if not files or not (root / 'scripts/v3_local_preview_entry.py').is_file():
        raise ValueError('Missing application source inputs')
    before = {relative: identity(root / relative) for relative in files}
    for relative in files:
        source, target = root / relative, snapshot / relative
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError('Source input escapes the checkout: ' + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError('Refusing to overwrite source snapshot: ' + relative)
        shutil.copyfile(source, target)
    after = {relative: identity(root / relative) for relative in selected_files(root)}
    copied = {relative: identity(snapshot / relative) for relative in files}
    if before != after or before != copied:
        raise RuntimeError('Application source changed while snapshotting; start a new build')
    manifest = {'schema': 'ptb-source-snapshot/1', 'files': copied}
    (snapshot / 'source-snapshot.json').write_text(json.dumps(manifest, indent=2) + '\n', 'utf8')
    return manifest


def validate(snapshot):
    snapshot = Path(snapshot)
    manifest = json.loads((snapshot / 'source-snapshot.json').read_text('utf8'))
    actual = {relative: identity(snapshot / relative) for relative in selected_files(snapshot)}
    if manifest.get('schema') != 'ptb-source-snapshot/1' or manifest['files'] != actual:
        raise RuntimeError('Frozen application source identity changed')
    return manifest


def import_paths(snapshot):
    return [str(Path(snapshot).resolve() / relative) for relative in IMPORT_ROOTS]
