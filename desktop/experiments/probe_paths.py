"""Narrow read-only asset resolution for the P01 custom URL scheme."""
from pathlib import Path
from urllib.parse import unquote


def resolve_asset(root: Path, url_path: str) -> Path:
    path = unquote(url_path)
    if not path.startswith('/') or path.startswith('//') or any(c in path for c in ('\\', ':', '\0')):
        raise ValueError('Invalid resource path')
    parts = path[1:].split('/')
    if any(part in ('..', '.', '') for part in parts):
        raise ValueError('Invalid resource path')
    resolved = (root / path[1:]).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError('Unknown resource')
    return resolved
