"""Refresh existing generated-reader asset pins from verified author and reader bytes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from manual.resources import safe_path


def refresh(root: Path, *, check=False):
    target = root / 'contracts/resource-manifest.json'
    registry = json.loads(target.read_text(encoding='utf-8'))
    book = json.loads((root / 'manual/project.json').read_text(encoding='utf-8'))
    assets = {asset['path']: asset for asset in book['assets']}
    prefix = 'frontend/public/manual/'
    changes = []
    for resource in registry['resources']:
        if not resource['path'].startswith(prefix):
            continue
        relative = resource['path'][len(prefix):]
        asset = assets.get(relative)
        if not asset:
            raise ValueError('Pinned reader asset is absent from the author project: ' + relative)
        source = safe_path(root / 'manual', relative)
        reader = safe_path(root, resource['path'])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if asset.get('sha256') is not None and asset['sha256'] != digest:
            raise ValueError('Author asset hash does not match its metadata: ' + relative)
        if hashlib.sha256(reader.read_bytes()).hexdigest() != digest:
            raise ValueError('Rebuild the reader before registering its asset: ' + relative)
        if resource['sha256'] != digest:
            resource['sha256'] = digest
            changes.append(resource['path'])
    if changes and not check:
        temporary = target.with_name(target.name + '.' + uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
        temporary.replace(target)
    return {'changed': changes, 'check': check, 'sourceAndGeneratedHashesEqual': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = refresh(ROOT, check=args.check)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(1 if args.check and result['changed'] else 0)
