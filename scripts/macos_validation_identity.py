"""Content identities recorded before and after actual Mac verification."""
import hashlib
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root):
    root = Path(root)
    paths = sorted(p for base in ('backend/src', 'packages/phonetic_core/src')
                   for p in (root / base).rglob('*.py'))
    if not paths:
        raise RuntimeError('Scientific source is missing')
    return {p.relative_to(root).as_posix(): sha256(p) for p in paths}


def runtime_hashes(root):
    root = Path(root)
    hashes = source_hashes(root)
    required = ('resources/research/reaper', 'scripts/macos_science_entry.py',
                'scripts/macos_validation_identity.py', 'scripts/verify_macos_runtime.py')
    paths = [root / name for name in required]
    paths += sorted((root / 'resources/vocal_tract/native/darwin-arm64').iterdir())
    for path in paths:
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = sha256(path)
        elif path in paths[:len(required)]:
            raise RuntimeError('Required runtime input missing: ' + str(path))
    return hashes


def require_identity(actual, expected, label):
    if not actual or actual != expected:
        raise RuntimeError(label + ' is not bound to the current source; rerun verification')
