"""Freeze licensed M19 boundary inputs without touching the active data directory.

Requests are bounded and never silently retried. Each successful response keeps
its request, retrieval time and SHA-256. OSM geometry uses one explicit historical
timestamp; Overpass areas are used for discovery only, not as polygon evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ENDPOINT = 'https://overpass-api.de/api/interpreter'
USER_AGENT = 'PhoneticToolbox-boundary-audit/3.0.5'


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def acquire(output, name, url, *, query=None, limit=80_000_000, timeout=65):
    """Only reuse cache bytes whose receipt and hash match this exact request."""
    path, receipt = output / name, output / (name + '.receipt.json')
    if path.exists() and receipt.exists():
        record = json.loads(receipt.read_text(encoding='utf-8'))
        if record.get('url') == url and record.get('sha256') == hashlib.sha256(path.read_bytes()).hexdigest():
            parsed = json.loads(path.read_bytes()) if name.endswith('.json') else {}
            if not parsed.get('remark') and ('status' not in parsed or parsed['status'] == 200):
                return record | {'cached': True}
    record = {'name': name, 'url': url, 'query': query,
              'retrievedAt': datetime.now(timezone.utc).isoformat()}
    try:
        with urlopen(Request(url, headers={'User-Agent': USER_AGENT}), timeout=timeout) as response:
            if int(response.headers.get('Content-Length', 0)) > limit:
                raise ValueError('Response exceeds acquisition size limit')
            data = response.read(limit + 1)
            # Only provenance-relevant headers. Do not retain cookies.
            record['headers'] = {k: response.headers[k] for k in ['Content-Type', 'Last-Modified', 'ETag'] if k in response.headers}
        if len(data) > limit:
            raise ValueError('Response exceeds acquisition size limit')
        if name.endswith('.json'):
            parsed = json.loads(data)
            if parsed.get('remark'):
                raise ValueError('Incomplete Overpass response: ' + parsed['remark'])
            if 'status' in parsed and parsed['status'] != 200:
                raise ValueError('Source API failure: ' + str(parsed.get('message', parsed['status'])))
            if 'elements' in parsed:
                record['elements'] = len(parsed['elements'])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.part')
        temporary.write_bytes(data)
        temporary.replace(path)
        record.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        save_json(receipt, record)
        print(name, record.get('elements', ''), len(data), flush=True)
    except Exception as error:
        record['error'] = str(error)
        save_json(output / (name + '.failure.json'), record)
        print(name, 'FAILED', str(error), flush=True)
    return record


def osm_query(output, name, tail, snapshot):
    query = f'[out:json][timeout:55][date:"{snapshot}"];' + tail
    return acquire(output, name, ENDPOINT + '?' + urlencode({'data': query}), query=query)


def flatten_mca(tree):
    result = []
    def visit(node, parent=None):
        if node.get('level'):
            raw = node['code']
            result.append({k: node.get(k) for k in ['name', 'level', 'type']} |
                          {'code': raw.ljust(6, '0') if re.fullmatch(r'\d{2}|\d{4}|\d{6}', raw) else None,
                           'sourceCode': raw, 'parentCode': parent})
        for child in node.get('children') or []:
            visit(child, node.get('code'))
    visit(tree['data'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--stage', choices=['official', 'osm-tags', 'osm-geometry'], required=True)
    parser.add_argument('--province-ids', type=int, nargs='*', help='Limit a resumed discovery run to specific OSM provinces')
    parser.add_argument('--relation-ids', type=int, nargs='*', help='Explicit approved geometry relation IDs')
    args = parser.parse_args()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', args.snapshot):
        parser.error('--snapshot must be a UTC ISO timestamp')
    output = args.output.resolve()
    active = Path(__file__).resolve().parents[1] / 'local-data'
    if output == active or active in output.parents:
        parser.error('Acquisition must use an independent staging directory outside local-data')
    output.mkdir(parents=True, exist_ok=True)
    results = []
    if args.stage == 'official':
        jobs = [('mca-current-tree.json', 'https://dmfw.mca.gov.cn/xzqh/getList?code=0&trimCode=true&maxLevel=3'),
                ('hk-official.json', 'https://www.had.gov.hk/psi/hong-kong-administrative-boundaries/hksar_18_district_boundary.json')]
        for name, url in jobs:
            results.append(acquire(output, name, url))
        if not any('error' in r for r in results):
            rows = flatten_mca(json.loads((output / 'mca-current-tree.json').read_text(encoding='utf-8')))
            save_json(output / 'mca-current-flat.json', rows)
            def version(province):
                code = province['sourceCode']
                url = 'https://dmfw.mca.gov.cn/xzqh/getCodeList?' + urlencode({
                    'tableName': 'Xzqh20251231', 'parentCode': code, 'placeCode': '',
                    'title': '', 'type': 3, 'pageNum': 1, 'pageSize': 5000})
                return acquire(output, f'mca-2025/{code}.json', url)
            for row in [r for r in rows if r['level'] == 1 and r['code'] and int(r['code']) < 700000]:
                result = version(row)
                results.append(result)
                if 'error' in result:
                    break
                time.sleep(2)
    elif args.stage == 'osm-tags':
        record = osm_query(output, 'osm-provinces.json', 'relation["ISO3166-2"~"^CN-"]["boundary"="administrative"];out tags;', args.snapshot)
        results.append(record)
        if 'error' not in record:
            provinces = json.loads((output / 'osm-provinces.json').read_text(encoding='utf-8'))['elements']
            selected = [e for e in provinces if e['tags'].get('admin_level') == '4' and (not args.province_ids or e['id'] in args.province_ids)]
            def children(element):
                rid = element['id']
                return osm_query(output, f'osm-tags/{rid}.json',
                    f'relation({rid});map_to_area;relation(area)["boundary"="administrative"]["admin_level"~"^[56]$"];out tags;', args.snapshot)
            for province in selected:
                result = children(province)
                results.append(result)
                if 'error' in result:
                    break
                time.sleep(3)
    else:
        if not args.relation_ids:
            parser.error('--relation-ids is required for geometry acquisition')
        # Small bounded batches keep requests restartable and retain exact OSM members.
        ids = sorted(set(args.relation_ids))
        for offset in range(0, len(ids), 8):
            batch = ids[offset:offset + 8]
            tail = 'relation(id:' + ','.join(map(str, batch)) + ');out meta geom;'
            results.append(osm_query(output, f'osm-geometry/{"-".join(map(str, batch))}.json', tail, args.snapshot))
            if 'error' in results[-1]:
                break
            time.sleep(3)
    save_json(output / f'{args.stage}-run.json', {'snapshot': args.snapshot, 'results': results})
    if any('error' in record for record in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
