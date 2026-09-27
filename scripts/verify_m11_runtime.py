"""Read-only existing-runtime inventory; no MFA import, activation, or installation."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    root = Path(root).resolve()
    if (root / 'env').is_dir():
        root = root / 'env'
    packages = []
    for path in sorted((root / 'conda-meta').glob('*.json')):
        p = json.loads(path.read_text(encoding='utf-8'))
        packages.append({k: p.get(k) for k in ('name', 'version', 'build', 'subdir', 'url', 'sha256', 'size', 'license', 'depends')})
    files = [p for p in root.rglob('*') if p.is_file() and not p.is_symlink()]
    return dict(schema='m11-inventory/1', packages=packages, files=len(files),
                installed_bytes=sum(p.stat().st_size for p in files),
                measured_archive_bytes=None, root=str(root),
                python_sha256=digest(root / 'python.exe' if (root / 'python.exe').exists() else root / 'bin/python'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', required=True)
    p.add_argument('--model')
    p.add_argument('--dictionary')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = inventory(a.runtime)
    for key in ('model', 'dictionary'):
        if value := getattr(a, key):
            path = Path(value)
            result[key] = dict(name=path.name, bytes=path.stat().st_size, sha256=digest(path))
            if key == 'model':
                with zipfile.ZipFile(path) as z:
                    result[key]['expanded_bytes'] = sum(i.file_size for i in z.infolist())
                    names = [n for n in z.namelist() if n.endswith('meta.json')]
                    result[key]['metadata'] = json.loads(z.read(names[0])) if len(names) == 1 else None
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(installed_bytes=result['installed_bytes'], packages=len(result['packages']),
                          files=result['files'], output=str(target)), ensure_ascii=False))


if __name__ == '__main__':
    main()
