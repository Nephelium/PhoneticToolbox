"""Explicit runtime-only omissions. Original source and notices are preserved."""
from pathlib import PurePosixPath

DEPENDENCY_POLICY = 'non-mfa-runtime-trim/2-multilingual'


def external_frontend_data(name):
    # The user's atlas extracts and curated tables are a separately delivered,
    # local research data pack. Never fold these into a public application EXE.
    return name == 'geolinguistics'


def dependency_exclusion(relative):
    """Exact omissions from scientific prefixes; Qt languages remain intact.

    Accept either a prefix-relative path or a packaged runtimes/<name> path.
    Never remove notices, public numpy.testing, or native compatibility helpers.
    """
    path = PurePosixPath(str(relative).replace('\\', '/'))
    parts = path.parts
    if parts[:1] == ('_internal',):
        parts = parts[1:]
    if len(parts) > 2 and parts[0] == 'runtimes' and parts[1] in {'egg', 'm05'}:
        parts = parts[2:]
    if any(p.lower().startswith(('license', 'licence', 'copying', 'copyright', 'notice')) for p in parts):
        return None
    if parts[:2] == ('Lib', 'idlelib'):
        return 'unused-python-idle-editor'
    if parts[:2] == ('Lib', 'pydoc_data'):
        return 'unused-interactive-help-topics'
    if parts[:4] == ('Lib', 'site-packages', 'cv2', 'data') and path.suffix.lower() == '.xml':
        return 'unused-opencv-haar-cascade-data'
    if parts[:3] == ('Lib', 'site-packages', 'numpy'):
        tail = parts[3:]
        if tail[:1] == ('testing',) or tail in {('_core', 'tests', '_natype.py'), ('_core', 'tests', '__init__.py')}:
            return None
        if 'tests' in tail and path.suffix.lower() not in {'.pyd', '.dll', '.so', '.dylib'}:
            return 'numpy-upstream-tests-with-runtime-support-retained'
    return None


def assert_dependency_policy(files):
    """Reject stale untrimmed caches and missing NumPy runtime support."""
    names = {str(name).replace('\\', '/').removeprefix('_internal/') for name in files}
    rejected = [name for name in sorted(names) if dependency_exclusion(name)]
    if rejected:
        raise ValueError('Runtime cache violates dependency policy; prepare a new tested stage: ' + rejected[0])
    for runtime in ('egg', 'm05'):
        base = 'runtimes/' + runtime + '/Lib/site-packages/'
        if base + 'numpy/__init__.py' in names:
            for support in ('numpy/testing/__init__.py', 'numpy/_core/tests/_natype.py'):
                if base + support not in names:
                    raise ValueError('Missing public NumPy runtime support: ' + base + support)


def exclusion(relative):
    path = PurePosixPath(str(relative).replace('\\', '/'))
    if path.parts[:2] == ('output', 'paper-reading') or 'paper-reading-content' in path.parts:
        return 'M18 papers are delivered separately by the content server'
    if path.parts[:2] == ('docs', 'manual'):
        return 'legacy-authoring-markdown; in-app book is frontend/dist/manual'
    if path.parts[:1] == ('contracts',) and path.suffix in {'.md', '.ts'}:
        return 'developer-documentation-or-generated-types; runtime JSON retained'
    return None
