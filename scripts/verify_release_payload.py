"""Read-only file hashes of an actual unpacked/installed Preview payload."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.payload.resolve()
    inventory = json.loads(args.inventory.read_text('utf8'))

    def verify(entry):
        name, expected = entry
        file = root / name
        if not file.is_file():
            return {'file': name, 'error': 'missing'}
        if file.stat().st_size != expected['size']:
            return {'file': name, 'error': 'size'}
        with file.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != expected['sha256']:
            return {'file': name, 'error': 'sha256', 'actual': digest}
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        errors = [item for item in pool.map(verify, inventory['files'].items()) if item]
    report = {'success': not errors, 'version': inventory['version'], 'payload': str(root),
              'expectedFileCount': inventory['fileCount'], 'verifiedBytes': inventory['expandedBytes'],
              'errors': errors, 'scope': 'All inventory files in the actual installed/unpacked tree; extra runtime files not asserted absent'}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', 'utf8')
    print(json.dumps({k: v for k, v in report.items() if k != 'errors'}, ensure_ascii=False), flush=True)
    assert not errors, errors[:10]


if __name__ == '__main__':
    main()
