"""Fetch the reviewed public map inputs into a task-local build directory.

No executable content is downloaded or run. Original bytes and their URLs/hashes
are retained; preparation is a separate, explicit command.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

COMMIT = 'ae72417fee2a63f453e6b2717e40655e87e26887'
BASE = f'https://raw.githubusercontent.com/GaryBikini/ChinaAdminDivisonSHP/{COMMIT}/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    entries = []
    for folder, name in [('2. Province', 'province'), ('3. City', 'city'), ('4. District', 'district')]:
        for ext in ['shp', 'dbf', 'shx', 'prj']:
            entries.append((f'{name}.{ext}', BASE + quote(f'{folder}/{name}.{ext}')))
    entries += [
        ('taiwan-county.zip', 'https://maps.nlsc.gov.tw/download/' + quote('縣市界線(TWD97經緯度).zip')),
        ('taiwan-town.zip', 'https://maps.nlsc.gov.tw/download/' + quote('鄉鎮市區界線(TWD97經緯度).zip')),
        ('world.geojson', 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson'),
    ]
    def fetch(item):
        name, url = item
        path = args.output / name
        try:
            if not path.exists():
                with urlopen(Request(url, headers={'User-Agent':'PhoneticToolbox-local-map-audit/1.0'}), timeout=60) as response:
                    data = response.read()
                if data[:20].lower().startswith(b'<!doctype'):
                    raise ValueError('HTML returned instead of map data')
                path.write_bytes(data)
            data = path.read_bytes()
            result = dict(name=name, url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            print(name, len(data), flush=True)
            return result
        except Exception as error:
            print(name, str(error), flush=True)
            return dict(name=name, url=url, error=str(error))
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, entries))
    (args.output / 'downloads.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    if any('error' in item for item in results):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
