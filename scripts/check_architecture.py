"""P02 static dependency and asset checks; not a security sandbox."""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from manual.resources import check_generated_manual, safe_path

LEGACY = re.compile(r'phonetic_toolbox|PhoneticToolbox_v2|dialect[_-]?(?:web|api)', re.I)
# Reviewed array-only OpenCV operations used by M09. No module namespace,
# wildcard, dynamic import, file codecs, camera/video or window API in core.
CORE_CV2_SYMBOLS = frozenset({'LINE_8', 'circle', 'getPerspectiveTransform', 'line', 'warpPerspective'})
DENIED = {
    'core': {'PyQt6', 'PySide6', 'fastapi', 'starlette', 'pydantic', 'ptb_api', 'ptb_desktop',
             'requests', 'httpx', 'http', 'urllib', 'socket', 'sqlite3', 'sqlalchemy', 'psycopg',
             'subprocess', 'sounddevice', 'cv2'},
    'backend': {'PyQt6', 'PySide6', 'ptb_desktop'},
    'desktop': {'ptb_api', 'fastapi', 'starlette', 'sqlalchemy', 'psycopg'},
}


def check_python(code, layer):
    errors = []
    if LEGACY.search(code):
        errors.append('legacy reference')
    if re.search(r'[A-Z]:[\\/]', code):
        errors.append('fixed machine path')
    tree = ast.parse(code)
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            if layer == 'core' and node.level == 0 and node.module == 'cv2':
                for item in node.names:
                    if item.name not in CORE_CV2_SYMBOLS:
                        errors.append('forbidden core OpenCV symbol: ' + item.name)
                continue
            names = [node.module or ''] + [item.name for item in node.names]
        elif isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Name) and func.id == '__import__') or (
                isinstance(func, ast.Attribute) and func.attr == 'import_module'):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    names = [node.args[0].value]
                else:
                    errors.append('dynamic import requires explicit architecture review')
        for name in names:
            if name.lstrip('.').split('.')[0] in DENIED[layer]:
                errors.append('forbidden import: ' + name)
    return errors


def check_frontend(code):
    errors = []
    if LEGACY.search(code) or re.search(r'[A-Z]:[\\/]', code):
        errors.append('legacy or machine path')
    imports = re.findall(r'''(?:from\s*|import\s*\(?\s*|require\s*\(\s*)["']([^"']+)["']''', code)
    for name in imports:
        parts = name.replace('\\', '/').split('/')
        if name.startswith('node:') or name.split('/')[0] in {'fs', 'child_process', 'electron', 'os', 'net'} or any(
                p in {'backend', 'desktop', 'packages', 'phonetic_core'} for p in parts):
            errors.append('forbidden frontend dependency: ' + name)
    return errors


def check_assets(root, manifest, source_ids):
    errors, registered = [], set()
    for item in manifest['resources']:
        relative = item['path']
        try:
            path = safe_path(root, relative)
        except (ValueError, OSError) as exc:
            errors.append('invalid resource path: ' + str(relative) + ': ' + str(exc))
            continue
        if relative in registered:
            errors.append('duplicate resource: ' + relative)
        registered.add(relative)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            errors.append('resource hash mismatch: ' + relative)
        if item.get('origin') == 'project':
            if item.get('source_id') or not item.get('description'):
                errors.append('invalid project resource provenance: ' + relative)
        elif item.get('source_id') not in source_ids:
            errors.append('unknown resource source: ' + relative)
    generated_paths = set()
    for declaration in manifest.get('generated_resources', []):
        relative = declaration.get('path', '')
        if declaration.get('kind') != 'manual-reader' or relative != 'frontend/public/manual':
            errors.append('unknown generated resource declaration: ' + relative)
            continue
        if relative in generated_paths:
            errors.append('duplicate generated resource declaration: ' + relative)
        generated_paths.add(relative)
        problems, files = check_generated_manual(root, declaration)
        errors.extend(problems)
        registered.update(files)
    for folder in ['frontend/public', 'frontend/src/assets', 'desktop/src/ptb_desktop/assets', 'packages/phonetic_core/src/phonetic_core/assets']:
        try:
            start = safe_path(root, folder)
            pending = [start] if start.is_dir() else []
            while pending:
                for file in pending.pop().iterdir():
                    relative = file.relative_to(root).as_posix()
                    try:
                        safe_path(root, relative)
                    except (ValueError, OSError) as exc:
                        errors.append('invalid resource path: ' + relative + ': ' + str(exc))
                        continue
                    if file.is_dir():
                        pending.append(file)
                    elif file.is_file() and relative not in registered:
                        errors.append('unregistered resource: ' + relative)
        except (ValueError, OSError) as exc:
            errors.append('resource scan failed: ' + folder + ': ' + str(exc))
    return errors


def main():
    errors = []
    for folder, layer in [('packages/phonetic_core/src', 'core'), ('backend/src', 'backend'), ('desktop/src', 'desktop')]:
        for path in (ROOT / folder).rglob('*.py'):
            for problem in check_python(path.read_text('utf-8'), layer):
                errors.append(f'{path.relative_to(ROOT)}: {problem}')
    for path in (ROOT / 'frontend/src').rglob('*'):
        if path.suffix in {'.ts', '.vue', '.js', '.css'}:
            errors += [f'{path.relative_to(ROOT)}: {e}' for e in check_frontend(path.read_text('utf-8'))]
    manifest = json.loads((ROOT / 'contracts/resource-manifest.json').read_text('utf-8'))
    registry = json.loads((ROOT / 'third_party/source-registry.json').read_text('utf-8'))
    errors += check_assets(ROOT, manifest, {s['id'] for s in registry['sources']})
    print(json.dumps({'errors': errors}, ensure_ascii=False))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
