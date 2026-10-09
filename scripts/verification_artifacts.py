"""Owned test scratch with persistent recipes, hashes and cleanup receipts.

Put generated/copied inputs, databases and large exports in ``scratch``.
Keep reports, logs and selected screenshots in ``out``. Close owned services
inside the context before exit. External --source paths are never adopted.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def no_links(path):
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            if item.is_symlink() or getattr(item.lstat(), 'st_file_attributes', 0) & 0x400:
                raise ValueError(f'Linked artifact path: {item}')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def inventory(path):
    """Record bytes without following links, including on Windows."""
    no_links(path)
    rows = []
    for directory, folders, files in os.walk(path, followlinks=False):
        for name in folders + files:
            no_links(Path(directory) / name)
        for name in sorted(files):
            file = Path(directory) / name
            with file.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            rows.append(dict(path=file.relative_to(path).as_posix(), bytes=file.stat().st_size, sha256=digest))
    return rows


def finish(out, marker, *, error=None):
    """Cleanup failure is visible and never replaces an original test failure."""
    try:
        rows = inventory(out / 'scratch')
        write_json(out / 'scratch-manifest.json', dict(schema='ptb-test-scratch-files/1', files=rows))
        marker.update(state='ready', test_outcome='failed' if error else 'completed',
                      error_type=type(error).__name__ if error else None)
        write_json(out / 'run.json', marker)
        if os.name != 'nt':
            raise RuntimeError('Automatic scratch cleanup currently supports Windows; review the manifest')
        result = subprocess.run(
            ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
             str(ROOT / 'scripts/cleanup_test_scratch.ps1'), '-RunDirectory', str(out)],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            creationflags=subprocess.CREATE_NO_WINDOW, check=False)
        if result.returncode:
            raise RuntimeError(f'Cleanup exited {result.returncode}: {result.stderr.strip()}')
        receipt = json.loads((out / 'cleanup.json').read_text('utf-8-sig'))
        if receipt['status'] != 'deleted':
            raise RuntimeError(receipt.get('reason', 'Scratch retained'))
    except Exception as exc:
        write_json(out / 'cleanup.json', dict(status='retained', path=str(out / 'scratch'), reason=str(exc)))
        print(f'Scratch retained; see {out / "cleanup.json"}', flush=True)


@contextmanager
def verification_run(suite, prefix, recipe, *, root=ROOT):
    if not all(re.fullmatch(r'[a-z0-9][a-z0-9-]*', value) for value in (suite, prefix)):
        raise ValueError('Use plain suite and run names')
    out = Path(root).absolute() / 'output/validation' / suite / (prefix + '-' + uuid4().hex)
    no_links(out)
    out.mkdir(parents=True, exist_ok=False)
    scratch = out / 'scratch'
    scratch.mkdir()
    marker = dict(schema='ptb-test-run/1', owner_pid=os.getpid(), state='active',
                  started_at=datetime.now(timezone.utc).isoformat(), recipe=recipe)
    script = Path(sys.argv[0])
    if script.is_file():
        marker['script'] = script.name
        marker['script_sha256'] = hashlib.sha256(script.read_bytes()).hexdigest()
    write_json(out / 'run.json', marker)
    error = None
    try:
        yield out, scratch
    except BaseException as exc:
        error = exc
        raise
    finally:
        finish(out, marker, error=error)
