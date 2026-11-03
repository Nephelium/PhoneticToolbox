"""Bind actual browser tests to an explicit, unmodified frontend candidate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


def frontend_identity(directory: Path) -> dict:
    files = []
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise AssertionError('Frontend verification input must not contain symlinks')
        if path.is_file():
            files.append({'path': path.relative_to(directory).as_posix(),
                          'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    assert any(row['path'] == 'index.html' for row in files)
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {'root': str(directory), 'treeSha256': hashlib.sha256(encoded).hexdigest(),
            'files': files}


def prepare_frontend(root: Path, scratch: Path, output: Path) -> tuple[Path, dict]:
    explicit = os.environ.get('PTB_VERIFY_FRONTEND_DIST')
    if explicit:
        directory = Path(explicit).resolve(strict=True)
        assert directory.is_absolute() and directory.is_dir()
    else:
        directory = scratch / 'frontend'
        with (output / 'frontend-build.log').open('w', encoding='utf-8') as log:
            subprocess.run(['node', str(root / 'frontend/node_modules/vite/bin/vite.js'),
                            'build', '--outDir', str(directory)], cwd=root / 'frontend',
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    identity = frontend_identity(directory)
    identity['explicitCandidate'] = bool(explicit)
    (output / 'frontend-input.json').write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    return directory, identity


def verify_frontend_unchanged(directory: Path, identity: dict, output: Path) -> None:
    actual = frontend_identity(directory)
    unchanged = actual['treeSha256'] == identity['treeSha256']
    (output / 'frontend-input-readback.json').write_text(json.dumps({
        'root': str(directory), 'treeSha256': actual['treeSha256'],
        'expectedTreeSha256': identity['treeSha256'], 'unchanged': unchanged,
        'explicitCandidate': identity['explicitCandidate'], 'fileCount': len(actual['files']),
    }, indent=2) + '\n', encoding='utf-8')
    assert unchanged, 'Frontend candidate changed during real browser verification'
