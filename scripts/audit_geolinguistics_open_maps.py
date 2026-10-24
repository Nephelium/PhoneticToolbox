"""Inspect an isolated boundary candidate without changing its runtime data."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--active', type=Path, required=True)
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--tools', type=Path, action='append', default=[])
    args = parser.parse_args()
    for tool in args.tools:
        sys.path.insert(0, str(tool.resolve()))
    from shapely.geometry import shape, Point
    from shapely.ops import unary_union
    from shapely.strtree import STRtree
    from package_geolinguistics_data import DATA_FILES
    candidate = args.candidate.resolve()
    if candidate == args.active.resolve() or 'local-data' in candidate.parts:
        raise ValueError('Audit an isolated candidate only')
    features = []
    for path in sorted((candidate / 'boundary-databases').glob('*.geojson')):
        features.extend(load(path)['features'])
    if not features:
        raise ValueError('Unsimplified per-source boundary databases are required')
    by_id = {f['id']: f for f in features}
    if len(by_id) != len(features):
        raise ValueError('Duplicate candidate identities')
    geometries = {f['id']: shape(f['geometry']) for f in features}
    mainland = load(candidate / 'mainland-audit.json')
    officials = [r for r in load(args.official / 'mca-current-flat.json') if r.get('code') and int(r['code']) < 700000]
    accepted = {r['code']: r for r in mainland['evidence']}
    # MCA's parallel county code identifies the same Kinmen entity that the
    # independently licensed NLSC county database represents. No duplicate
    # district polygon, software ID or changed parent is manufactured.
    kinmen = by_id.get('city:TW-09020')
    cross_regional = []
    if kinmen and kinmen['properties']['name'] == '金門縣' and kinmen['properties']['source'] == 'nlsc-official':
        cross_regional.append({'officialCode': '350527', 'officialName': '金门县', 'representedBy': kinmen['id'],
                               'sourceName': '金門縣', 'nativeCode': '09020', 'sourceId': 'M19-NLSC',
                               'method': 'explicit same-county correspondence between separate administrative coding systems',
                               'scope': 'geographic coverage only; original hierarchy and classification rows are preserved'})
    cross_codes = {r['officialCode'] for r in cross_regional}
    per_province = []
    for province in [r for r in officials if r['level'] == 1]:
        rows = [r for r in officials if r['code'][:2] == province['code'][:2]]
        missing = [r for r in rows if r['code'] not in accepted and r['code'] not in cross_codes]
        per_province.append({'code': province['code'], 'name': province['name'],
                             'expectedByLevel': dict(Counter(r['level'] for r in rows)),
                             'matchedByLevel': dict(Counter(r['level'] for r in rows if r['code'] in accepted)),
                             'crossRegionalCodes': [r['code'] for r in rows if r['code'] in cross_codes], 'missing': missing,
                             'duplicateOfficialMatches': [code for code, count in Counter(r['code'] for r in mainland['evidence'] if r['code'][:2] == province['code'][:2]).items() if count > 1]})
    duplicates, overlapping = [], []
    district_features = [f for f in features if f['properties']['level'] == 'district']
    district_geometries = [geometries[f['id']] for f in district_features]
    tree = STRtree(district_geometries)
    for i, geom in enumerate(district_geometries):
        for value in tree.query(geom, predicate='intersects'):
            j = int(value)
            if j <= i:
                continue
            a, b = district_features[i], district_features[j]
            intersection = geom.intersection(district_geometries[j])
            if intersection.area <= 1e-12:
                continue
            entry = {'a': a['id'], 'aName': a['properties']['name'], 'b': b['id'], 'bName': b['properties']['name'],
                     'sameParent': a['properties']['parent'] == b['properties']['parent'],
                     'areaDegrees2': intersection.area, 'fractionOfA': intersection.area / geom.area,
                     'fractionOfB': intersection.area / district_geometries[j].area}
            overlapping.append(entry)
            if geom.equals(district_geometries[j]):
                duplicates.append(entry)
    children = defaultdict(list)
    for f in features:
        if f['properties']['parent'] != '100000':
            children[f['properties']['parent']].append(f)
    hierarchy = []
    for identifier, items in children.items():
        if identifier not in geometries:
            hierarchy.append({'parent': identifier, 'reason': 'missing-parent', 'children': [f['id'] for f in items]})
            continue
        parent = geometries[identifier]
        union = unary_union([geometries[f['id']] for f in items])
        hierarchy.append({'parent': identifier, 'children': len(items), 'parentAreaDegrees2': parent.area,
                          'uncoveredParentAreaDegrees2': parent.difference(union).area,
                          'childAreaOutsideParentDegrees2': union.difference(parent).area,
                          'scope': 'source disagreement audit, not automatic clipping or acceptance'})
    probes = []
    for label, lon, lat in [('海南海口', 110.33, 20.02), ('舟山定海', 122.10, 30.02), ('澎湖', 119.575, 23.57),
                            ('金门', 118.317, 24.432), ('马祖南竿', 119.93, 26.15), ('东沙岛', 116.728, 20.704),
                            ('太平岛', 114.366, 10.377), ('香港大屿山', 113.94, 22.27), ('澳门路氹', 113.563, 22.145)]:
        point = Point(lon, lat)
        hits = [district_features[int(i)]['id'] for i in tree.query(point, predicate='intersects')]
        probes.append({'label': label, 'coordinate': [lon, lat], 'districtHits': sorted(hits),
                       'interpretation': 'geographic spot-check; no administrative identity inferred from proximity'})
    files = []
    for name in DATA_FILES:
        path = candidate / name if (candidate / name).exists() else args.active / name
        files.append({'path': name, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    report = {'status': 'review-required', 'mainlandExpected': len(officials), 'mainlandDirectMatches': len(accepted),
              'crossRegionalCoverage': cross_regional, 'perProvince': per_province,
              'unresolvedOfficialEntities': [r for r in officials if r['code'] not in accepted and r['code'] not in cross_codes],
              'invalidUnsimplifiedGeometry': [identifier for identifier, geom in geometries.items() if geom.is_empty or not geom.is_valid],
              'duplicateDistrictGeometry': duplicates, 'districtOverlaps': sorted(overlapping, key=lambda r: r['areaDegrees2'], reverse=True),
              'hierarchyAreaAudit': hierarchy, 'geographicProbes': probes, 'runtimeFiles': files,
              'runtimeTotalBytes': sum(f['bytes'] for f in files), 'runtimeMaximumFileBytes': max(f['bytes'] for f in files),
              'sourceScopeCaution': 'Exact names, identity chains and count coverage do not certify statutory boundary accuracy. Source overlaps, parent disagreements, code changes and regional scopes remain explicit.'}
    (candidate / 'quality-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'mainlandExpected', 'mainlandDirectMatches', 'crossRegionalCoverage', 'unresolvedOfficialEntities', 'invalidUnsimplifiedGeometry', 'runtimeTotalBytes', 'runtimeMaximumFileBytes']}, ensure_ascii=False, indent=2))
    print('District overlap pairs:', len(overlapping), 'duplicate district geometries:', len(duplicates), flush=True)


if __name__ == '__main__':
    main()
