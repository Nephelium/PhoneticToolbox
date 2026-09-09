"""P02 static dependency and asset checks; not a security sandbox."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = re.compile(r'phonetic_toolbox|PhoneticToolbox_v2|dialect[_-]?(?:web|api)', re.I)
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
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            errors.append('resource outside workspace: ' + relative)
            continue
        registered.add(relative)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            errors.append('resource hash mismatch: ' + relative)
        if item['source_id'] not in source_ids:
            errors.append('unknown resource source: ' + relative)
    for folder in ['frontend/public', 'frontend/src/assets', 'desktop/src/ptb_desktop/assets', 'packages/phonetic_core/src/phonetic_core/assets']:
        for file in (root / folder).rglob('*'):
            if file.is_file() and file.relative_to(root).as_posix() not in registered:
                errors.append('unregistered resource: ' + file.relative_to(root).as_posix())
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
