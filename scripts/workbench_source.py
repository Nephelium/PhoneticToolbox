"""Bind development processes to one checkout before importing application code."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ('packages/phonetic_core/src', 'backend/src', 'desktop/src')
PACKAGES = {
    'phonetic_core': 'packages/phonetic_core/src/phonetic_core',
    'ptb_api': 'backend/src/ptb_api',
    'ptb_worker': 'backend/src/ptb_worker',
    'ptb_desktop': 'desktop/src/ptb_desktop',
}
RUNTIMES = {
    'PTB_EGG_PYTHON': '.venv/m03-compatible/python.exe',
    'PTB_M05_PYTHON': '.venv/m05/Scripts/python.exe',
}
MODULES = tuple(f'M{i:02d}' for i in range(1, 19))


def bind_sources(root=ROOT):
    """Fail on a mixed import cache; never hide it by replacing sys.modules."""
    root = Path(root).resolve()
    expected = {name: root / relative for name, relative in PACKAGES.items()}
    for name, path in expected.items():
        if not (path / '__init__.py').is_file():
            raise RuntimeError(f'Missing current source package: {name}: {path}')
    for name, module in tuple(sys.modules.items()):
        package = name.partition('.')[0]
        if package not in expected or module is None:
            continue
        origin = getattr(module, '__file__', None)
        locations = [origin] if origin else list(getattr(module, '__path__', ()))
        if any(not Path(path).resolve().is_relative_to(expected[package]) for path in locations):
            raise RuntimeError(f'Application code was already imported outside this checkout: {name}')
    sources = [str(root / path) for path in SOURCE_ROOTS]
    sys.path[:] = sources + [path for path in sys.path if path not in sources]
    os.environ['PYTHONPATH'] = os.pathsep.join(sources)
    os.environ['PYTHONNOUSERSITE'] = '1'
    os.environ['PYTHONSAFEPATH'] = '1'
    os.environ.pop('PYTHONHOME', None)
    importlib.invalidate_caches()
    origins = {}
    for name, path in expected.items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != path / '__init__.py':
            raise RuntimeError(f'Current source binding failed: {name}')
        origins[name] = str(Path(spec.origin).resolve())
    return origins


def configure(root=ROOT, *, component_root=None):
    """Pin project interpreters; MFA stays external and explicitly configured."""
    root = Path(root).resolve()
    sources = bind_sources(root)
    bindings = {}
    for key, relative in RUNTIMES.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f'Required project runtime is missing: {path}')
        bindings[key] = str(path)
    os.environ.update(bindings)
    if component_root:
        path = Path(component_root).resolve()
        if not path.is_dir():
            raise RuntimeError(f'MFA component directory does not exist: {path}')
        os.environ['PTB_M11_COMPONENT_ROOT'] = str(path)
    required = ('PyQt6.QtWebEngineWidgets', 'numpy', 'scipy', 'parselmouth', 'pandas',
                'cv2', 'soundfile', 'sounddevice', 'docx', 'xlrd', 'openpyxl')
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError('Incomplete host runtime; use Start-Research-Workbench.ps1. Missing: ' + ', '.join(missing))
    return dict(schema='ptb-development-sources/1', root=str(root), python=sys.executable,
                sources=sources, runtimes=bindings, frontend=str(root / 'frontend/dist'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', choices=('home', *MODULES), default='home')
    parser.add_argument('--component-root')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    options = parser.parse_args(argv)
    report = configure(component_root=options.component_root)
    report['module'] = options.module
    if options.check_only:
        print(json.dumps(report, ensure_ascii=False))
        return 0
    from start_m01_workbench import main as start
    return start(['--module', options.module] + (['--prepare-only'] if options.prepare_only else []))


if __name__ == '__main__':
    raise SystemExit(main())
