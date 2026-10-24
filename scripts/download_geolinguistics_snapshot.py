"""Download one fixed Geofabrik snapshot, validating size and upstream MD5.

The output is an explicitly chosen task directory. Existing data is never
overwritten. SHA-256 and source timestamps are recorded for later reproduction.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

SOURCES = {
    'guangdong': ('https://download.geofabrik.de/asia/china/guangdong-261022.osm.pbf', 174037667),
    'china': ('https://download.geofabrik.de/asia/china-261022.osm.pbf', 1613578208),
}
SNAPSHOT = '2026-10-22T20:21:33Z'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region', choices=SOURCES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    url, expected_size = SOURCES[args.region]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / url.rsplit('/', 1)[-1]
    part = path.with_suffix(path.suffix + '.part')
    receipt = path.with_suffix(path.suffix + '.receipt.json')
    if path.exists() or part.exists() or receipt.exists():
        parser.error('Target or partial file already exists; inspect its receipt before resuming')
    headers = {'User-Agent': 'PhoneticToolbox-boundary-audit/3.0.5'}
    with urlopen(Request(url + '.md5', headers=headers), timeout=30) as response:
        upstream_md5 = response.read(1024).decode('ascii').strip()
    expected_md5 = upstream_md5.split()[0]
    record = {'url': url, 'snapshot': SNAPSHOT, 'expectedBytes': expected_size,
              'upstreamMd5': upstream_md5, 'retrievedAt': datetime.now(timezone.utc).isoformat(),
              'license': 'ODbL-1.0', 'attribution': '© OpenStreetMap contributors; extract by Geofabrik'}
    md5, sha, size = hashlib.md5(), hashlib.sha256(), 0
    try:
        with urlopen(Request(url, headers=headers), timeout=90) as response, part.open('xb') as stream:
            record['headers'] = {k: response.headers[k] for k in ['Content-Length', 'Last-Modified', 'ETag'] if k in response.headers}
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > expected_size:
                    raise ValueError('Snapshot exceeds published byte length')
                stream.write(chunk)
                md5.update(chunk)
                sha.update(chunk)
                if size % (64 * 1024 * 1024) == 0:
                    print(f'{args.region}: {size}/{expected_size}', flush=True)
        if size != expected_size or md5.hexdigest() != expected_md5:
            raise ValueError('Snapshot size or upstream MD5 mismatch')
        part.replace(path)
        record.update(bytes=size, md5=md5.hexdigest(), sha256=sha.hexdigest(), status='verified-download')
    except Exception as error:
        record.update(bytes=size, error=str(error), status='failed-download')
        raise
    finally:
        receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
