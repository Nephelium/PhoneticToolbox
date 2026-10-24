"""Acquire an isolated, unchanged GeoAtlas areas_v3 study candidate.

Uses only the official selector's all.json tree and documented URL suffixes.
No runtime integration, coordinate conversion, topology repair or publication.
The upstream selector labels this version 2021.5, not a current-year boundary set.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from threading import Lock
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

BASE = 'https://geo.datav.aliyun.com/areas_v3/bound/'
PAGE = 'https://datav.aliyun.com/portal/school/atlas/area_selector'
DOC = 'https://help.aliyun.com/zh/datav/datav-7-0/user-guide/datav-geoatlas-widgets/'
SCRIPT_SHA = 'ff114f649f490ab62137088aae8f32594378dae233f2cd5ed122156182b19b5d'
MUNICIPAL = {110000, 120000, 310000, 500000, 810000, 820000}
PATTERN = re.compile(r'(?:all|\d{6}(?:_full(?:_city|_district)?|_boundary)?)\.json\Z')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8')


def atomic(path, raw):
    temporary = path.with_suffix(path.suffix + '.part')
    for parent in (path, temporary, *path.parents):
        if parent.is_symlink() or parent.exists() and getattr(parent.lstat(), 'st_file_attributes', 0) & 0x400:
            raise ValueError('Refusing a linked output path: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('wb') as stream:
        stream.write(raw)
        stream.flush()
    temporary.replace(path)


def receipt_path(output, name):
    return output / (name.removesuffix('.json') + '.receipt.json')


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


class Acquisition:
    def __init__(self, output, timeout=25, attempts=3, delay=.35, limit=40_000_000):
        self.output, self.timeout, self.attempts = output, timeout, attempts
        self.delay, self.limit = delay, limit
        self.network_attempts = 0
        self.counter_lock = Lock()

    def cached(self, name):
        target, receipt = self.output / name, receipt_path(self.output, name)
        if receipt.exists():
            record = json.loads(receipt.read_bytes())
            if record['url'] != BASE + name:
                raise ValueError('Receipt URL mismatch: ' + name)
            if record.get('status') == 200:
                if not target.is_file() or digest(target.read_bytes()) != record['sha256']:
                    raise ValueError('Cached bytes do not match receipt: ' + name)
                return record | {'reuse': True}
            if record.get('status') == 404:
                return record | {'reuse': True}
        elif target.exists():
            raise ValueError('Existing raw input has no verified receipt: ' + name)
        return None

    def seed(self, directory, records):
        for source in records:
            name = source['name'] + '.json'
            if not PATTERN.fullmatch(name) or source['url'] != BASE + name:
                raise ValueError('Seed is not an official selector URL')
            if self.cached(name):
                continue
            record = {'url': source['url'], 'name': name, 'version': 'areas_v3',
                      'copiedAt': now(), 'seedDirectory': str(directory), 'seedReceipt': source}
            if source.get('downloaded') or source.get('status') == 200:
                raw = (directory / name).read_bytes()
                if digest(raw) != source['sha256'] or len(raw) != source['bytes']:
                    raise ValueError('Seed SHA/size mismatch: ' + name)
                record.update(status=200, bytes=len(raw), sha256=digest(raw))
                atomic(self.output / name, raw)
            elif source.get('status') == 404:
                record.update(status=404, responseSha256=source.get('responseSha256'),
                              responseBytes=source.get('responseBytes'), note='Reused original HTTP404 receipt; no repeat GET')
            else:
                raise ValueError('Unsupported seed status: ' + name)
            atomic(receipt_path(self.output, name), encoded(record))

    def acquire(self, name):
        if not PATTERN.fullmatch(name):
            raise ValueError('Unlisted URL form: ' + name)
        old = self.cached(name)
        if old:
            return old
        record = {'url': BASE + name, 'name': name, 'version': 'areas_v3', 'attempts': []}
        for attempt in range(self.attempts):
            time.sleep(self.delay if not attempt else min(2 ** attempt, 8))
            detail = {'startedAt': now(), 'number': attempt + 1}
            raw = None
            with self.counter_lock:
                self.network_attempts += 1
            try:
                opener = build_opener(NoRedirect())
                request = Request(record['url'], headers={'User-Agent': 'PhoneticToolbox-GeoAtlas-study/3.0.5', 'Accept': 'application/json'})
                try:
                    response = opener.open(request, timeout=self.timeout)
                except HTTPError as error:
                    response = error
                with response:
                    status = response.code
                    detail.update(status=status, finalUrl=response.geturl(), headers={
                        k: response.headers[k] for k in ('Content-Type', 'Content-Length', 'Last-Modified', 'ETag') if k in response.headers})
                    if int(response.headers.get('Content-Length', 0)) > self.limit:
                        raise ValueError('Response exceeds byte limit')
                    deadline = time.monotonic() + self.timeout * 2
                    chunks, size = [], 0
                    while True:
                        chunk = response.read(min(65536, self.limit + 1 - size))
                        if not chunk:
                            break
                        chunks.append(chunk)
                        size += len(chunk)
                        if size > self.limit or time.monotonic() > deadline:
                            raise ValueError('Response exceeds byte/time limit')
                    raw = b''.join(chunks)
                detail.update(bytes=len(raw), sha256=digest(raw), finishedAt=now())
                record['attempts'].append(detail)
                record.update(status=status, retrievedAt=detail['finishedAt'])
                if status == 200:
                    parsed = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant: ' + value)))
                    if name == 'all.json':
                        if not isinstance(parsed, list):
                            raise ValueError('Official tree must be an array')
                    elif not isinstance(parsed, dict) or parsed.get('type') != 'FeatureCollection' or not isinstance(parsed.get('features'), list):
                        raise ValueError('Official geometry must be a FeatureCollection')
                    atomic(self.output / name, raw)
                    record.update(bytes=len(raw), sha256=digest(raw))
                    atomic(receipt_path(self.output, name), encoded(record))
                    print(json.dumps({'file': name, 'status': status, 'bytes': len(raw)}), flush=True)
                    return record
                # Keep the original error response, including 404, for diagnosis.
                atomic(self.output / (name + '.http' + str(status) + '.body'), raw)
                record.update(responseBytes=len(raw), responseSha256=digest(raw))
                atomic(receipt_path(self.output, name), encoded(record))
                if status not in (408, 429, 500, 502, 503, 504):
                    break
            except (URLError, OSError, TimeoutError) as error:
                detail.update(error=type(error).__name__ + ': ' + str(error), finishedAt=now())
                record['attempts'].append(detail)
                record.update(status=None, error=detail['error'])
                atomic(receipt_path(self.output, name), encoded(record))
            except (ValueError, TypeError) as error:
                if raw is not None:
                    atomic(self.output / (name + '.invalid-response.body'), raw)
                    record.update(responseBytes=len(raw), responseSha256=digest(raw))
                detail.update(error=str(error), finishedAt=now())
                if not record['attempts'] or record['attempts'][-1] is not detail:
                    record['attempts'].append(detail)
                record.update(status=None, error=detail['error'])
                atomic(receipt_path(self.output, name), encoded(record))
                break
        print(json.dumps({'file': name, 'status': record.get('status'), 'error': record.get('error')}), flush=True)
        return record

    def batch(self, names):
        # Fixed maximum of two simultaneous requests, including retries.
        with ThreadPoolExecutor(max_workers=2) as pool:
            return list(pool.map(self.acquire, sorted(set(names))))


def tree_index(tree):
    result = {}
    for node in tree:
        code = node.get('adcode')
        if not isinstance(code, int) or not re.fullmatch(r'\d{6}', str(code)) or code in result:
            raise ValueError('Invalid/duplicate code in official tree')
        if node.get('level') not in ('country', 'province', 'city', 'district'):
            raise ValueError('Unexpected official tree level')
        result[code] = node
    if 100000 not in result:
        raise ValueError('Official tree has no national root')
    for node in tree:
        seen, parent = {node['adcode']}, node.get('parent')
        while parent is not None:
            if parent in seen or parent not in result:
                raise ValueError('Missing/cyclic official parent')
            seen.add(parent)
            parent = result[parent].get('parent')
    return result


def aggregate_plan(index):
    names = {'100000_full_city.json'}
    for code, node in index.items():
        if node['level'] != 'province' or code == 710000:
            continue  # Taiwan's selector exposes only the already-seeded outline.
        names.add(str(code) + '_full.json')
        if code not in MUNICIPAL:
            names.add(str(code) + '_full_district.json')
    return sorted(names)


def file_features(output):
    for receipt in sorted(output.glob('*.receipt.json')):
        record = json.loads(receipt.read_bytes())
        if record.get('status') != 200 or record['name'] == 'all.json':
            continue
        path = output / record['name']
        raw = path.read_bytes()
        if digest(raw) != record['sha256']:
            raise ValueError('Raw file changed after acquisition: ' + path.name)
        data = json.loads(raw)
        yield path.name, data


def coverage(output, index):
    found, extras, mismatches, parent_mismatches = set(), [], [], []
    for name, data in file_features(output):
        for position, feature in enumerate(data['features']):
            properties = feature.get('properties') or {}
            code, geometry = properties.get('adcode'), feature.get('geometry') or {}
            if not isinstance(code, int) or code not in index:
                extras.append({'file': name, 'feature': position, 'adcode': code, 'name': properties.get('name'), 'geometryType': geometry.get('type')})
                continue
            if geometry.get('type') in ('Polygon', 'MultiPolygon') and geometry.get('coordinates'):
                found.add(code)
            if properties.get('level') and properties['level'] != index[code]['level']:
                mismatches.append({'file': name, 'feature': position, 'adcode': code, 'sourceLevel': properties['level'], 'treeLevel': index[code]['level']})
            source_parent = properties.get('parent')
            source_parent = source_parent.get('adcode') if isinstance(source_parent, dict) else source_parent
            if source_parent is not None and source_parent != index[code]['parent']:
                parent_mismatches.append({'file': name, 'feature': position, 'adcode': code, 'sourceParent': source_parent, 'treeParent': index[code]['parent']})
    missing = [node for code, node in index.items() if code != 100000 and code not in found]
    return {'expected': dict(Counter(n['level'] for n in index.values())),
            'covered': dict(Counter(index[code]['level'] for code in found)),
            'missing': missing, 'extraAndSymbols': extras, 'levelMismatches': mismatches, 'parentMismatches': parent_mismatches}


def audit(output, index):
    from shapely import __version__ as shapely_version, geos_version_string
    from shapely.geometry import shape
    from shapely.validation import explain_validity
    files, layers, variants = {}, defaultdict(list), defaultdict(set)
    for name, data in file_features(output):
        rows = []
        for position, feature in enumerate(data['features']):
            props, geometry = feature.get('properties') or {}, feature.get('geometry')
            code = props.get('adcode')
            row = {'file': name, 'feature': position, 'adcode': code, 'name': props.get('name'),
                   'sourceLevel': props.get('level'), 'sourceParent': props.get('parent'),
                   'geometryType': (geometry or {}).get('type'),
                   'geometrySha256': digest(json.dumps(geometry, sort_keys=True, separators=(',', ':')).encode('utf8'))}
            try:
                value = shape(geometry)
                row.update(valid=bool(value.is_valid), validity=explain_validity(value), empty=bool(value.is_empty),
                           bounds=[float(v) if math.isfinite(v) else None for v in value.bounds],
                           parts=len(value.geoms) if hasattr(value, 'geoms') else 1)
            except Exception as error:
                row.update(valid=False, validity=type(error).__name__ + ': ' + str(error), empty=None)
            rows.append(row)
            layer = index[code]['level'] if isinstance(code, int) and code in index else 'symbols-and-other'
            layers[layer].append(row)
            if isinstance(code, int) and code in index:
                variants[code].add(row['geometrySha256'])
        files[name] = {'features': len(rows), 'types': dict(Counter(r['geometryType'] for r in rows)),
                       'crs': data.get('crs'), 'invalid': [r for r in rows if not r['valid']]}
    for level, rows in layers.items():
        atomic(output / 'layers' / (level + '.index.json'), encoded(rows))
    report = {'shapely': shapely_version, 'geos': geos_version_string,
              'operation': 'Read-only shape/is_valid/explain_validity; no conversion, clipping, union or make_valid',
              'files': files, 'layerFeatures': {key: len(rows) for key, rows in layers.items()},
              'layerUniqueCodes': {key: len({r['adcode'] for r in rows if isinstance(r['adcode'], int)}) for key, rows in layers.items()},
              'multipleRawGeometriesPerCode': [{'adcode': code, 'variants': len(values)} for code, values in variants.items() if len(values) > 1]}
    atomic(output / 'geometry-audit.json', encoded(report))
    return report


def main():
    started = now()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True, help='Existing six raw official probes plus provenance.json')
    parser.add_argument('--tree-probe', type=Path, help='Optional already-fetched official all.json + receipt')
    parser.add_argument('--page-evidence', type=Path, required=True, help='Existing official atlas.js and widgets.html cache')
    parser.add_argument('--shapely-path', type=Path, help='Existing task-local Shapely package directory; never installs')
    parser.add_argument('--offline', action='store_true', help='Reaudit existing receipts without requests')
    args = parser.parse_args()
    output = args.output.absolute()
    for parent in (output, *output.parents):
        if parent.exists() and (parent.is_symlink() or getattr(parent.lstat(), 'st_file_attributes', 0) & 0x400):
            parser.error('Output cannot contain junctions/symlinks')
    repository = Path(__file__).resolve().parents[1]
    if output.resolve() == repository or repository in output.resolve().parents:
        parser.error('Raw acquisition must be outside the product repository')
    output.mkdir(parents=True, exist_ok=True)
    script = (args.page_evidence / 'atlas.js').read_bytes()
    if digest(script) != SCRIPT_SHA:
        parser.error('Official selector script changed; review the new URL rules before fetching')
    text = script.decode('utf8')
    version = re.search(r'\{value:"areas_v3",label:"areas_v3",time:"([^"]+)"\}', text)
    if not version or version[1] != '2021.5' or '"all.json"' not in text:
        parser.error('Expected official selector evidence is missing')
    evidence = {'page': PAGE, 'documentation': DOC, 'scriptUrl': 'https://g.alicdn.com/datav/datav-portal/0.46.1/atlas.js',
                'scriptSha256': digest(script), 'version': 'areas_v3', 'pageVersionTime': version[1],
                'pageVersionTimeMeaning': 'Official selector metadata; not a claim of 2026 administrative currency',
                'source': 'DataV GeoAtlas / 高德开放平台', 'useNotice': '数据来自高德开放平台，仅供学习和交流使用。',
                'dataLicense': 'Official learning/exchange notice retained; data is not licensed as project MIT code',
                'rules': {'tree': BASE + 'all.json', 'national': ['', '_full', '_full_city'],
                          'ordinaryProvince': ['', '_full', '_full_district'], 'municipalityAndHKMacau': ['', '_full'],
                          'taiwan': [''], 'city': ['', '_full'], 'district': ['']},
                'rawPolicy': 'Preserve original bytes and every symbol/island; no OSM/NLSC data mixed; no coordinate conversion/topology repair'}
    widget = args.page_evidence / 'widgets.html'
    if widget.exists():
        evidence['officialDocCacheSha256'] = digest(widget.read_bytes())
    atomic(output / 'source-evidence.json', encoded(evidence))
    acquisition = Acquisition(output)
    seeds = json.loads((args.probe / 'provenance.json').read_bytes())['files']
    acquisition.seed(args.probe, seeds)
    if args.tree_probe:
        raw = (args.tree_probe / 'all.json').read_bytes()
        tree_receipt = json.loads((args.tree_probe / 'all.receipt.json').read_bytes())
        if tree_receipt['url'] != BASE + 'all.json' or digest(raw) != tree_receipt['sha256']:
            parser.error('Tree probe SHA/URL mismatch')
        acquisition.seed(args.tree_probe, [tree_receipt | {'name': 'all', 'downloaded': True}])
    if not args.offline:
        acquisition.acquire('all.json')
    if not acquisition.cached('all.json'):
        parser.error('Official tree unavailable')
    index = tree_index(json.loads((output / 'all.json').read_bytes()))
    plan = aggregate_plan(index)
    atomic(output / 'acquisition-plan.json', encoded({'aggregateFiles': plan, 'maxConcurrency': 2, 'attemptsPerURL': 3, 'timeoutSeconds': 25, 'delaySeconds': .35}))
    if not args.offline:
        acquisition.batch(plan)
        # Tree-driven fallback: city aggregate first, only then exact missing leaves.
        missing = coverage(output, index)['missing']
        parents = {node['parent'] for node in missing if node['level'] == 'district' and index[node['parent']]['level'] == 'city'}
        acquisition.batch([str(code) + '_full.json' for code in parents])
        missing = coverage(output, index)['missing']
        acquisition.batch([str(node['adcode']) + '.json' for node in missing])
    result = coverage(output, index)
    atomic(output / 'coverage.json', encoded(result))
    if args.shapely_path:
        sys.path.insert(0, str(args.shapely_path.resolve()))
    geometry = audit(output, index)
    records = [json.loads(p.read_bytes()) for p in sorted(output.glob('*.receipt.json'))]
    inventory = {'status': 'raw_candidate_for_review', 'createdAt': now(), 'evidence': evidence,
                 'files': records, 'successfulFiles': sum(r.get('status') == 200 for r in records),
                 'missingFiles404': [r['name'] for r in records if r.get('status') == 404],
                 'failedFiles': [r['name'] for r in records if r.get('status') not in (200, 404)],
                 'networkAttemptsThisRun': acquisition.network_attempts,
                 'rawSourceSetSha256': digest(encoded({r['name']: r['sha256'] for r in records if r.get('status') == 200})),
                 'coverage': {key: value for key, value in result.items() if key not in ('extraAndSymbols', 'levelMismatches', 'parentMismatches')},
                 'geometry': {'fileCount': len(geometry['files']), 'invalidFeatureOccurrences': sum(len(v['invalid']) for v in geometry['files'].values()),
                              'layerUniqueCodes': geometry['layerUniqueCodes']},
                 'limitations': ['2021.5 upstream selector version; not certified current administrative divisions.',
                                 'Coverage is against this exact official tree, not proof of every national boundary/island or coordinate registration.',
                                 'Invalid geometries remain untouched for the next import/completeness review.',
                                 'No active runtime, research coordinates, service or public download endpoint changed.']}
    atomic(output / 'source-inventory.json', encoded(inventory))
    run = {'startedAt': started, 'finishedAt': now(), 'offline': args.offline,
           'networkAttempts': acquisition.network_attempts, 'scriptSha256': digest(Path(__file__).read_bytes()),
           'inventorySha256': digest((output / 'source-inventory.json').read_bytes()),
           'rawSourceSetSha256': inventory['rawSourceSetSha256']}
    history = output / 'acquisition-runs.jsonl'
    atomic(history, (history.read_bytes() if history.exists() else b'') + json.dumps(run, ensure_ascii=False).encode('utf8') + b'\n')
    print(json.dumps({k: inventory[k] for k in ('successfulFiles', 'missingFiles404', 'failedFiles', 'networkAttemptsThisRun', 'rawSourceSetSha256', 'coverage', 'geometry')}, ensure_ascii=False, indent=2), flush=True)
    return 1 if inventory['failedFiles'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
