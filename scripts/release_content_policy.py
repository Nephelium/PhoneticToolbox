"""Explicit runtime-only omissions. Original source and notices are preserved."""
from pathlib import PurePosixPath


def exclusion(relative):
    path = PurePosixPath(str(relative).replace('\\', '/'))
    if path.parts[:2] == ('output', 'paper-reading') or 'paper-reading-content' in path.parts:
        return 'M18 papers are delivered separately by the content server'
    if path.parts[:2] == ('docs', 'manual'):
        return 'legacy-authoring-markdown; in-app book is frontend/dist/manual'
    if path.parts[:1] == ('contracts',) and path.suffix in {'.md', '.ts'}:
        return 'developer-documentation-or-generated-types; runtime JSON retained'
    return None
