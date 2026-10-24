"""Build and audit an isolated licensed boundary candidate. Never replaces data.

Mainland identity requires exact official names, the correct administrative
level, non-conflicting explicit codes, and polygon containment or an explicit
frozen source subarea chain. Geometry inconsistencies remain separately audited.
Point matching only uses polygon intersection. Ambiguity stays explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

HK_NAMES = ['中西區', '灣仔區', '東區', '南區', '油尖旺區', '深水埗區', '九龍城區', '黃大仙區', '觀塘區',
            '荃灣區', '屯門區', '元朗區', '北區', '大埔區', '西貢區', '沙田區', '葵青區', '離島區']
HK_DISPLAY = ['中西区', '湾仔', '东区', '南区', '油尖旺', '深水埗', '九龙城', '黄大仙', '观塘',
              '荃湾', '屯门', '元朗', '北区', '大埔', '西贡', '沙田', '葵青', '离岛']
MACAU_NAMES = ['花地玛堂区', '花王堂区', '望德堂区', '大堂区', '风顺堂区', '嘉模堂区', '路氹填海区', '圣方济各堂区']
CODE_KEYS = ['ref:admin:CN', 'ref:admin:CN:mca', 'ref:admin:CN:nbs', 'ref:GB/T 2260']


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def names(tags):
    return {tags[key].strip() for key in ['name', 'name:zh', 'name:zh-Hans', 'official_name', 'official_name:zh']
            if tags.get(key) and tags[key].strip()}


def explicit_codes(tags):
    result = set()
    for key in CODE_KEYS:
        for token in tags.get(key, '').split(';'):
            token = token.strip()
            if re.fullmatch(r'\d{2}|\d{4}|\d{6}', token):
                result.add(token.ljust(6, '0'))
            elif re.fullmatch(r'\d{6}(?:000|000000)', token):
                result.add(token[:6])
    return result


def covered_fraction(child, parent):
    if child.area <= 0:
        return 0.0
    if parent.covers(child):
        return 1.0
    return max(0.0, min(1.0, 1.0 - child.difference(parent).area / child.area))


def contained(child, parent):
    # Only a numerical overlay tolerance, not an allowance for missing territory.
    return covered_fraction(child, parent) >= 1.0 - 1e-8


def feature(level, code, name, parent, province, geometry, source, **extra):
    return {'type': 'Feature', 'id': f'{level}:{code}', 'geometry': geometry,
            'properties': {'id': f'{level}:{code}', 'code': code, 'name': name, 'level': level,
                           'parent': parent, 'province': province, 'source': source, **extra}}


def geometry_structure(geometry):
    polygons = list(geometry.geoms) if geometry.geom_type == 'MultiPolygon' else [geometry]
    return {'parts': len(polygons), 'holes': sum(len(p.interiors) for p in polygons),
            'vertices': sum(len(p.exterior.coords) + sum(len(r.coords) for r in p.interiors) for p in polygons)}


def apply_geometry_derivations(osm, correspondence):
    """Apply only reviewed, version-pinned unions. Keep source tags and lineage."""
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    by_id = {f['properties']['osmId']: f for f in osm}
    if len(by_id) != len(osm):
        raise ValueError('Duplicate source geometry IDs')
    replacements, audit = {}, []
    for patch in (correspondence or {}).get('geometryDerivations', []):
        target = by_id[patch['target']['osmId']]
        if target['properties']['osmVersion'] != patch['target']['osmVersion'] or target['properties']['tags']['name'] != patch['target']['sourceName']:
            raise ValueError('Reviewed derivation target version/name changed')
        inputs = []
        for item in patch['components']:
            source = by_id[item['osmId']]
            if source['properties']['osmVersion'] != item['osmVersion'] or source['properties']['tags']['name'] != item['sourceName']:
                raise ValueError('Reviewed derivation component version/name changed')
            inputs.append(shape(source['geometry']))
        geometry = unary_union(inputs)
        if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in {'Polygon', 'MultiPolygon'}:
            raise ValueError('Reviewed derivation does not create a valid polygon')
        result = copy.deepcopy(target)
        result['geometry'] = mapping(geometry)
        result['properties']['originalSourceTags'] = copy.deepcopy(target['properties']['tags'])
        if patch.get('officialNameCorrection'):
            for key in ['name', 'name:zh', 'name:zh-Hans', 'official_name']:
                result['properties']['tags'][key] = patch['officialName']
        result['properties']['geometryDerivation'] = copy.deepcopy(patch)
        replacements[patch['target']['osmId']] = result
        original = shape(target['geometry'])
        audit.append({'officialCode': patch['officialCode'], 'officialName': patch['officialName'], 'operation': 'reviewed source polygon union',
                      'sourceComponents': patch['components'], 'evidence': patch['evidence'], 'urls': patch['urls'],
                      'addedAreaDegrees2': geometry.difference(original).area, 'removedAreaDegrees2': original.difference(geometry).area,
                      'sourceGeometryUnchanged': True, 'derivedGeometryValid': geometry.is_valid})
    return [replacements.get(f['properties']['osmId'], f) for f in osm], audit


def mainland(osm, officials, province_codes=None, correspondence=None, relations=None):
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    official = {r['code']: r for r in officials if r.get('code') and int(r['code']) < 700000
                and (not province_codes or r['code'][:2] in province_codes)}
    by_name = defaultdict(list)
    for row in official.values():
        by_name[(row['level'], row['name'])].append(row)
    geometries = {f['properties']['osmId']: shape(f['geometry']) for f in osm}
    reviewed = {r['osmId']: r for r in (correspondence or {}).get('entries', [])}
    source_relations = {r['id']: r for r in (relations or {}).get('relations', [])}
    def subarea_path(parent, child, depth=2):
        if depth <= 0 or parent not in source_relations:
            return None
        children = [m['ref'] for m in source_relations[parent]['members'] if m['type'] == 'r' and m['role'] == 'subarea']
        if child in children:
            return [parent, child]
        for middle in children:
            tail = subarea_path(middle, child, depth - 1)
            if tail:
                return [parent, *tail]
        return None
    accepted, evidence, pending, used = {}, [], [], set()
    for official_level, osm_level in [(1, '4'), (2, '5'), (3, '6')]:
        proposed = defaultdict(list)
        for f in osm:
            props, tags = f['properties'], f['properties']['tags']
            if tags.get('admin_level') != osm_level:
                continue
            rid = props['osmId']
            source_names = names(tags)
            identity_review = reviewed.get(rid)
            if identity_review and identity_review['officialCode'] not in official:
                continue
            if identity_review:
                native = identity_review['nativeCode']
                if (props['osmVersion'] != identity_review['osmVersion'] or tags.get('name') != identity_review['sourceName']
                        or tags.get(native['key']) != native['value']):
                    raise ValueError(f'Reviewed OSM identity changed: {rid}')
                row = official.get(identity_review['officialCode'])
                if row is None or row['name'] != identity_review['officialName'] or row['level'] != official_level:
                    raise ValueError(f'Reviewed official identity changed: {rid}')
                source_names = source_names | {identity_review['officialName']}
            candidates = {r['code']: r for name in source_names for r in by_name[(official_level, name)]}
            codes = explicit_codes(tags)
            matched, conflicts = [], []
            for code, row in candidates.items():
                reviewed_legacy = bool(identity_review and identity_review['officialCode'] == code
                                       and codes == set(identity_review.get('legacyCodes', [])))
                if codes and code not in codes and not reviewed_legacy:
                    conflicts.append({'code': code, 'reason': 'explicit-code-conflict', 'sourceCodes': sorted(codes)})
                    continue
                if official_level == 1:
                    if not tags.get('ISO3166-2', '').startswith('CN-'):
                        continue
                    matched.append(row)
                    continue
                province = accepted.get(code[:2] + '0000')
                if not province:
                    continue
                province_path = subarea_path(province['sourceFeature']['properties']['osmId'], rid)
                if not contained(geometries[rid], province['geometry']) and not province_path:
                    continue
                # Same-name counties in one province need their actual prefecture
                # polygon, not a nearest-centre or name-suffix heuristic.
                same_province = [r for r in candidates.values() if r['code'][:2] == code[:2]]
                if len(same_province) > 1:
                    parent = accepted.get(row['parentCode'].ljust(6, '0'))
                    if not parent or (not contained(geometries[rid], parent['geometry'])
                                      and not subarea_path(parent['sourceFeature']['properties']['osmId'], rid, 1)):
                        continue
                matched.append(row)
            if len(matched) == 1:
                proposed[matched[0]['code']].append((f, matched[0]))
            elif candidates:
                pending.append({'osmId': rid, 'name': tags.get('name'), 'adminLevel': osm_level,
                                'candidateCodes': sorted(candidates), 'conflicts': conflicts,
                                'reason': 'ambiguous-or-not-contained' if not conflicts else 'code-conflict'})
        for code, choices in proposed.items():
            if len(choices) != 1:
                pending.append({'code': code, 'osmIds': [f['properties']['osmId'] for f, _ in choices], 'reason': 'duplicate-relations'})
                continue
            f, row = choices[0]
            rid, tags = f['properties']['osmId'], f['properties']['tags']
            accepted[code] = {'row': row, 'geometry': geometries[rid], 'sourceFeature': f}
            used.add(rid)
            evidence.append({'code': code, 'name': row['name'], 'officialLevel': row['level'],
                             'osmId': rid, 'osmVersion': f['properties']['osmVersion'],
                             'sourceNames': sorted(names(f['properties'].get('originalSourceTags', tags))), 'sourceCodes': sorted(explicit_codes(tags)),
                             'method': 'exact-official-name+admin-level+province-containment',
                             'parentCode': row['parentCode'],
                             **({'reviewedCorrespondence': reviewed[rid]} if rid in reviewed else {})})
            if f['properties'].get('geometryDerivation'):
                evidence[-1]['geometryDerivation'] = f['properties']['geometryDerivation']
                evidence[-1]['method'] = 'reviewed-administrative-scope-union+official-identity'
            if row['level'] > 1:
                province = accepted[code[:2] + '0000']
                fraction = covered_fraction(geometries[rid], province['geometry'])
                evidence[-1]['provinceCoveredFraction'] = fraction
                evidence[-1]['sourceSubareaPath'] = subarea_path(province['sourceFeature']['properties']['osmId'], rid)
                if fraction < 1 - 1e-8:
                    evidence[-1]['method'] = 'exact-official-identity+explicit-source-subarea-chain'
                    evidence[-1]['geometryStatus'] = 'outside-province-polygon-review-required; original geometry retained'
    features = []
    for code, item in accepted.items():
        row, geom = item['row'], item['geometry']
        level = {1: 'province', 2: 'city', 3: 'district'}[row['level']]
        province = 'province:' + code[:2] + '0000'
        parent_code = row['parentCode'].ljust(6, '0')
        if level == 'province':
            parent = '100000'
        elif level == 'city':
            parent = province
        elif len(row['parentCode']) == 4:
            parent = 'city:' + parent_code
        elif code[:2] in {'11', '12', '31', '50'}:
            parent = 'city:' + code[:4] + '00'
        else:
            parent = 'city:' + code
        source_properties = item['sourceFeature']['properties']
        features.append(feature(level, code, row['name'], parent, province, mapping(geom), 'osm-20261022',
                                sourceId='M19-OSM', officialLevel=row['level'], osmId=source_properties['osmId'],
                                osmVersion=source_properties['osmVersion'], geometrySnapshot=source_properties.get('snapshot'),
                                nativeCode={k: source_properties['tags'][k] for k in CODE_KEYS + ['division_code']
                                            if k in source_properties['tags']},
                                codeAuthority='MCA live tree retrieved 2026-10-24', codeYear=None,
                                codeYearNote='Effective code year is not declared by the current-tree API response',
                                **({'geometryDerivation': source_properties['geometryDerivation']} if source_properties.get('geometryDerivation') else {})))
        if level == 'city' and not any(r['level'] == 3 and r['parentCode'].ljust(6, '0') == code for r in official.values()):
            # Dongguan, Zhongshan, Jiayuguan and Danzhou have no county-level
            # children. Keep the existing point-association/display area without
            # inventing a county in the official completeness denominator.
            features.append(feature('district', code, row['name'], 'city:' + code, province, mapping(geom),
                                    'osm-20261022', officialLevel=2, displayAggregate=True))
        if level == 'district' and len(row['parentCode']) == 2 and code[:2] not in {'11', '12', '31', '50'}:
            features.append(feature('city', code, row['name'], province, province, mapping(geom), 'osm-20261022',
                                    officialLevel=3, displayAggregate=True))
    # Preserve the existing display grouping of municipalities explicitly. These
    # unions are labelled display aggregates and are not claimed as prefectures.
    municipal_names = {'110100': '北京城区', '120100': '天津城区', '310100': '上海城区',
                       '500100': '重庆城区', '500200': '重庆郊县'}
    for code, label in municipal_names.items():
        children = [x for c, x in accepted.items() if x['row']['level'] == 3 and c.startswith(code[:4])]
        expected = [r for r in official.values() if r['level'] == 3 and r['code'].startswith(code[:4])]
        if children and len(children) == len(expected):
            province = 'province:' + code[:2] + '0000'
            features.append(feature('city', code, label, province, province, mapping(unary_union([x['geometry'] for x in children])),
                                    'osm-20261022', officialLevel=None, displayAggregate=True))
    audit = {'expected': len(official), 'accepted': len(accepted),
             'acceptedLevels': dict(Counter(x['row']['level'] for x in accepted.values())),
             'missing': [r for code, r in official.items() if code not in accepted],
             'pending': pending, 'evidence': evidence,
             'unusedOsm': [{'osmId': f['properties']['osmId'], 'name': f['properties']['tags'].get('name'),
                            'adminLevel': f['properties']['tags'].get('admin_level')}
                           for f in osm if f['properties']['osmId'] not in used]}
    return features, audit


def regional(official_dir, osm):
    import shapefile
    from pyproj import Transformer
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union, transform
    result, audit = [], {'taiwan': {}, 'hongKong': {}, 'macau': {}}
    transform_tw = Transformer.from_crs(3824, 4326, always_xy=True)
    town_geometries, town_props = {}, {}
    def read_zip(filename, stem):
        z = zipfile.ZipFile(official_dir / filename)
        reader = shapefile.Reader(shp=io.BytesIO(z.read(stem + '.shp')), shx=io.BytesIO(z.read(stem + '.shx')),
                                 dbf=io.BytesIO(z.read(stem + '.dbf')), encoding='utf-8')
        return [(r.record.as_dict(), transform(transform_tw.transform, shape(r.shape.__geo_interface__))) for r in reader.shapeRecords()]
    county_rows = read_zip('taiwan-county.zip', 'COUNTY_MOI_1090820')
    town_rows = read_zip('taiwan-town.zip', 'TOWN_MOI_1120317')
    supplements = read_zip('taiwan-town.zip', 'Town_Majia_Sanhe')
    for props, geom in town_rows:
        code = props['TOWNCODE']
        if code in town_geometries:
            raise ValueError('Duplicate official Taiwan town code')
        town_geometries[code], town_props[code] = geom, props
    additions = []
    for props, geom in supplements:
        code = props['TOWNCODE']
        if code not in town_props or town_props[code]['TOWNNAME'] != props['TOWNNAME']:
            raise ValueError('Taiwan supplemental polygon has no exact code/name target')
        additions.append({'code': code, 'name': props['TOWNNAME'], 'source': 'Town_Majia_Sanhe',
                          'outsideBaseAreaDegrees2': geom.difference(town_geometries[code]).area})
        town_geometries[code] = unary_union([town_geometries[code], geom])
    for props, geom in county_rows:
        code = 'TW-' + props['COUNTYCODE']
        result.append(feature('city', code, props['COUNTYNAME'], 'province:710000', 'province:710000', mapping(geom), 'nlsc-official',
                              officialLevel='county-city'))
    result.append(feature('province', '710000', '台湾省', '100000', 'province:710000',
                          mapping(unary_union([geom for _, geom in county_rows])), 'nlsc-official', displayAggregate=True))
    for code, geom in town_geometries.items():
        props = town_props[code]
        result.append(feature('district', 'TW-' + code, props['TOWNNAME'], 'city:TW-' + props['COUNTYCODE'],
                              'province:710000', mapping(geom), 'nlsc-official', officialLevel='township'))
    audit['taiwan'] = {'countyCities': len(county_rows), 'townships': len(town_rows), 'supplements': additions,
                       'sourceCrs': 'EPSG:3824', 'targetCrs': 'EPSG:4326', 'operation': transform_tw.description,
                       'accuracyMetres': transform_tw.accuracy}
    hk = load(official_dir / 'hk-official.json')['features']
    found = set()
    for source in hk:
        name = source['properties']['地區']
        if name not in HK_NAMES or name in found:
            raise ValueError('Unrecognised or duplicate official Hong Kong district')
        found.add(name)
        number = HK_NAMES.index(name)
        result.append(feature('district', str(810001 + number), HK_DISPLAY[number], 'city:810000', 'province:810000',
                              source['geometry'], 'hk-had-official', sourceName=name,
                              officialDistrictId=source['properties']['地區號碼'], officialLevel='district'))
    if len(found) != 18:
        raise ValueError('Hong Kong must include all 18 named official districts')
    # The official district dataset supplies the aggregate. Do not mix an OSM
    # outer polygon into these same official districts.
    hk_union = mapping(unary_union([shape(f['geometry']) for f in hk]))
    for level, parent in [('province', '100000'), ('city', 'province:810000')]:
        result.append(feature(level, '810000', '香港特别行政区', parent, 'province:810000', hk_union,
                              'hk-had-official', displayAggregate=True))
    audit['hongKong'] = {'districts': len(found), 'identityMethod': 'explicit official district name to existing display code crosswalk'}
    mo_sources = {}
    for f in osm:
        tags = f['properties']['tags']
        for name in names(tags) & set(MACAU_NAMES):
            if tags.get('admin_level') == '6':
                if name in mo_sources:
                    raise ValueError('Duplicate Macau district relation')
                mo_sources[name] = f
    for number, name in enumerate(MACAU_NAMES):
        if name not in mo_sources:
            continue
        f = mo_sources[name]
        result.append(feature('district', str(820001 + number), name, 'city:820000', 'province:820000',
                              f['geometry'], 'osm-20261022', osmId=f['properties']['osmId'], officialLevel='parish-or-reclamation-area'))
    mo_outer = [f for f in osm if f['properties']['tags'].get('ISO3166-2') == 'CN-MO']
    if len(mo_outer) == 1:
        for level, parent in [('province', '100000'), ('city', 'province:820000')]:
            result.append(feature(level, '820000', '澳门特别行政区', parent, 'province:820000',
                                  mo_outer[0]['geometry'], 'osm-20261022', displayAggregate=(level == 'city')))
    audit['macau'] = {'districts': len(mo_sources), 'missing': sorted(set(MACAU_NAMES) - mo_sources.keys()),
                       'outerBoundaryCount': len(mo_outer)}
    return result, audit


def audit_associations(features, active):
    from shapely.geometry import shape, Point
    from shapely.strtree import STRtree
    by_id = {f['id']: f for f in features}
    districts = [f for f in features if f['properties']['level'] == 'district']
    geometries = [shape(f['geometry']) for f in districts]
    tree = STRtree(geometries)
    source_sites = load(active / 'sites.json')
    sites = copy.deepcopy(source_sites)
    changed_sites = []
    for site in sites:
        before = site.get('admin')
        point = Point(site['lon'], site['lat'])
        hits = sorted(districts[int(i)]['id'] for i in tree.query(point, predicate='intersects'))
        site['adminCandidates'] = hits
        site['admin'] = hits[0] if len(hits) == 1 else None
        if before != site['admin']:
            changed_sites.append({'id': site['id'], 'name': site['name'], 'before': before,
                                  'after': site['admin'], 'candidates': hits, 'lon': site['lon'], 'lat': site['lat']})
    source_classification = load(active / 'classification.json')
    classification = copy.deepcopy(source_classification)
    old_index = {r['id']: r for r in load(active / 'map-index.json')}
    changed_rows, retained, unmatched = [], 0, Counter()
    for sheet in classification['sheets']:
        for row in sheet['rows']:
            before = row.get('admin')
            # Preserve an earlier reviewed identity only if the same code AND
            # exact name survives. Changed names/codes require explicit review.
            if before in by_id and before in old_index and old_index[before]['name'] == by_id[before]['properties']['name']:
                row['match'] = 'same-code-and-name-open-boundary'
                retained += 1
            else:
                row['admin'] = None
                row['match'] = 'open-boundary-correspondence-pending'
                if before:
                    changed_rows.append({'id': row['id'], 'before': before, 'after': None, 'values': row['values'][:6]})
            if not row.get('admin'):
                unmatched[sheet['id']] += 1
    site_value_changes = sum({k: v for k, v in before.items() if k not in {'admin', 'adminCandidates'}} !=
                            {k: v for k, v in after.items() if k not in {'admin', 'adminCandidates'}}
                            for before, after in zip(source_sites, sites))
    classification_value_changes = sum({k: v for k, v in before.items() if k not in {'admin', 'match'}} !=
                                      {k: v for k, v in after.items() if k not in {'admin', 'match'}}
                                      for original_sheet, new_sheet in zip(source_classification['sheets'], classification['sheets'])
                                      for before, after in zip(original_sheet['rows'], new_sheet['rows']))
    coordinate_changes = sum((a['lon'], a['lat']) != (b['lon'], b['lat']) for a, b in zip(source_sites, sites))
    if site_value_changes or classification_value_changes or coordinate_changes:
        raise ValueError('Boundary association altered an original research value')
    audit = {'sites': len(sites), 'sitesMatched': sum(bool(s['admin']) for s in sites),
             'sitesMultiple': sum(len(s['adminCandidates']) > 1 for s in sites), 'changedSites': changed_sites,
             'classificationIdentityRetained': retained, 'classificationUnmatchedBySheet': dict(unmatched),
             'changedClassificationRows': changed_rows, 'pointMethod': 'unique polygon intersection; no nearest-centre fallback',
             'siteValueChanges': site_value_changes, 'coordinateChanges': coordinate_changes,
             'classificationValueChanges': classification_value_changes}
    return sites, classification, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--osm', type=Path, required=True)
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--active', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tools', type=Path, action='append', default=[])
    parser.add_argument('--province-codes', nargs='*')
    parser.add_argument('--correspondence', type=Path, default=Path(__file__).resolve().parents[1] / 'third_party/patches/geolinguistics-osm-20261022.json')
    parser.add_argument('--relations', type=Path, help='Frozen OSM relation metadata for explicit subarea-chain identity evidence')
    parser.add_argument('--additional-osm', type=Path, action='append', default=[], help='Explicitly audited supplemental source geometry')
    args = parser.parse_args()
    for tools in args.tools:
        sys.path.insert(0, str(tools.resolve()))
    from shapely.geometry import shape, mapping
    output = args.output.resolve()
    active = args.active.resolve()
    if output == active or active in output.parents or output.exists():
        parser.error('Choose a new isolated candidate directory')
    from package_geolinguistics_data import DATA_FILES
    source_hashes = {name: hashlib.sha256((active / name).read_bytes()).hexdigest() for name in DATA_FILES}
    osm = load(args.osm)['features']
    source_by_id = {f['properties']['osmId']: f for f in osm}
    reused_source_ids = []
    for extra in args.additional_osm:
        for f in load(extra)['features']:
            identifier = f['properties']['osmId']
            if identifier in source_by_id:
                previous = source_by_id[identifier]
                if previous['properties'] != f['properties'] or not shape(previous['geometry']).equals(shape(f['geometry'])):
                    raise ValueError('Supplement conflicts with an existing frozen source geometry')
                reused_source_ids.append(identifier)
            else:
                osm.append(f)
                source_by_id[identifier] = f
    correspondence = load(args.correspondence) if args.correspondence.exists() else None
    osm, derivation_audit = apply_geometry_derivations(osm, correspondence)
    features, mainland_audit = mainland(osm, load(args.official / 'mca-current-flat.json'), args.province_codes,
                                       correspondence,
                                       load(args.relations) if args.relations else None)
    regional_features, regional_audit = regional(args.official, osm)
    features.extend(regional_features)
    if len({f['id'] for f in features}) != len(features):
        raise ValueError('Duplicate candidate identity')
    invalid = [f['id'] for f in features if shape(f['geometry']).is_empty or not shape(f['geometry']).is_valid]
    # Report invalid inputs, do not silently make_valid or drop polygon parts.
    if invalid:
        raise ValueError('Invalid candidate polygons: ' + ', '.join(invalid))
    sites, classification, associations = audit_associations(features, active)
    index, hierarchy_issues = [], []
    by_id = {f['id']: f for f in features}
    for f in features:
        props, geom = f['properties'], shape(f['geometry'])
        point = geom.representative_point()
        # Representative points serve label placement only, never association.
        index.append(props | {'center': [point.x, point.y], 'extent': list(geom.bounds)})
        if props['parent'] != '100000':
            parent = by_id.get(props['parent'])
            if parent is None:
                hierarchy_issues.append({'id': f['id'], 'parent': props['parent'], 'reason': 'missing-parent'})
            else:
                fraction = covered_fraction(geom, shape(parent['geometry']))
                if fraction < 1 - 1e-8:
                    hierarchy_issues.append({'id': f['id'], 'parent': props['parent'], 'reason': 'outside-parent', 'coveredFraction': fraction})
    output.mkdir(parents=True)
    save(output / 'map-index.json', index)
    display_audit = {}
    for level, tolerance in [('province', .008), ('city', .003), ('district', .001)]:
        layer = []
        quality = []
        for f in features:
            if f['properties']['level'] == level:
                g = copy.deepcopy(f)
                original = shape(f['geometry'])
                display = original.simplify(tolerance, preserve_topology=True)
                before, after = geometry_structure(original), geometry_structure(display)
                if not display.is_valid or display.is_empty or before['parts'] != after['parts'] or before['holes'] != after['holes']:
                    raise ValueError('Display simplification lost a polygon part or hole: ' + f['id'])
                quality.append({'id': f['id'], 'source': before, 'display': after, 'valid': True})
                g['geometry'] = mapping(display)
                layer.append(g)
        path = output / 'maps' / (level + '.json')
        save(path, {'type': 'FeatureCollection', 'features': layer})
        display_audit[level] = {'bytes': path.stat().st_size, 'sourceVertices': sum(r['source']['vertices'] for r in quality),
                                'displayVertices': sum(r['display']['vertices'] for r in quality), 'features': quality,
                                'lostParts': 0, 'lostHoles': 0, 'invalidDisplayGeometries': 0}
    # Independent regional databases carry their own licenses. These unsimplified
    # databases supply the geometry used for association and derivative-data
    # access; linguistic observations and personal edits are never included.
    databases = []
    for source in sorted({f['properties']['source'] for f in features}):
        path = output / 'boundary-databases' / (source + '.geojson')
        group = [f for f in features if f['properties']['source'] == source]
        save(path, {'type': 'FeatureCollection', 'features': group})
        databases.append({'source': source, 'path': str(path.relative_to(output)).replace('\\', '/'), 'features': len(group),
                          'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    save(output / 'display-geometry-audit.json', display_audit)
    save(output / 'sites.json', sites)
    save(output / 'classification.json', classification)
    save(output / 'mainland-audit.json', mainland_audit)
    save(output / 'regional-audit.json', regional_audit)
    save(output / 'association-audit.json', associations)
    save(output / 'derivation-audit.json', derivation_audit)
    summary = {'status': 'candidate-review-required', 'mainlandExpected': mainland_audit['expected'],
               'identicalSupplementalSourceIdsReused': reused_source_ids,
               'mainlandMatched': mainland_audit['accepted'], 'mainlandMissing': len(mainland_audit['missing']),
               'layerCounts': dict(Counter(f['properties']['level'] for f in features)),
               'regional': regional_audit, 'hierarchyIssues': hierarchy_issues,
               'sites': associations['sites'], 'sitesMatched': associations['sitesMatched'],
               'siteChanges': len(associations['changedSites']),
               'classificationUnmatchedBySheet': associations['classificationUnmatchedBySheet'],
               'classificationChanges': len(associations['changedClassificationRows']),
               'boundaryDatabases': databases,
               'runtimeMapBytes': sum(display_audit[level]['bytes'] for level in display_audit),
               'unchangedActiveInputSha256': source_hashes,
               'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [args.osm, *args.additional_osm, args.correspondence, args.official / 'mca-current-flat.json', active / 'sites.json', active / 'classification.json']},
               'displaySimplification': {'province': .008, 'city': .003, 'district': .001,
                                        'units': 'degrees', 'pointAssociationsUseUnsimplifiedGeometry': True}}
    if any(hashlib.sha256((active / name).read_bytes()).hexdigest() != digest for name, digest in source_hashes.items()):
        raise ValueError('An active source input changed while the candidate was being built')
    save(output / 'candidate-audit.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {'inputs', 'hierarchyIssues'}}, ensure_ascii=False, indent=2))
    print('Hierarchy issues:', len(hierarchy_issues), '(full list in candidate-audit.json)')


if __name__ == '__main__':
    main()
