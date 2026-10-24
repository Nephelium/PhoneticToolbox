"""Build an isolated 16-file GeoAtlas runtime candidate from frozen inputs.

Research values remain unchanged. Display coordinates use the existing, explicitly
inferred GCJ inverse once; reviewed WGS84 supplements bypass that inverse.
No active resources, frontend, services or public downloads are modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import inspect
import json
from pathlib import Path
import re
import sys

from normalize_geolinguistics_datav import compact, statistics
from package_geolinguistics_data import DATA_FILES, BOUNDARY_BINDINGS, data_manifest
from prepare_geolinguistics import gcj_inverse
from prepare_geolinguistics_datav import atomic, digest, encoded, tree_index

SOURCE = 'datav-areas-v3-inferred-gcj-wgs84'
IMMUTABLE = ('catalog.json', 'observations.json', 'paper.json', 'main.f64',
             'type-0-0.f64', 'type-0-1.f64', 'type-1-0.f64', 'type-1-1.f64')


def checked(path, sha):
    if path.is_symlink() or digest(path.read_bytes()) != sha:
        raise ValueError('Frozen input SHA mismatch: ' + str(path))


class CoordinateAdapter:
    def __init__(self):
        self.cache = {}
        self.positions = 0

    def geometry(self, value):
        result = copy.deepcopy(value)
        def transform(coords):
            if coords and isinstance(coords[0], (int, float)):
                if len(coords) != 2:
                    raise ValueError('Only explicit 2D source coordinates are supported')
                key = tuple(coords)
                self.positions += 1
                if key not in self.cache:
                    self.cache[key] = gcj_inverse(*key)
                return list(self.cache[key])
            return [transform(child) for child in coords]
        if result['type'] == 'GeometryCollection':
            result['geometries'] = [self.geometry(g) for g in result['geometries']]
        else:
            result['coordinates'] = transform(result['coordinates'])
        return result


def ancestry(code, nodes):
    values, seen = [], set()
    while code is not None:
        if code in seen or code not in nodes:
            raise ValueError('Invalid official ancestry')
        seen.add(code);values.append(code);code = nodes[code]['parent']
    return values


def runtime_id(code, nodes):
    return '100000' if code == 100000 else nodes[code]['level'] + ':' + str(code)


def choose_deepest(hits, nodes):
    """Eliminate only actual hit ancestors, retaining incomparable conflicts."""
    values = set(hits)
    excluded = {ancestor for code in values for ancestor in ancestry(code, nodes)[1:] if ancestor in values}
    return sorted(values - excluded), sorted(excluded)


def associate_sites(source_sites, geometries, nodes):
    from shapely import STRtree
    from shapely.geometry import Point
    codes = sorted(geometries)
    tree = STRtree([geometries[code] for code in codes])
    sites, audit = copy.deepcopy(source_sites), []
    for before, site in zip(source_sites, sites):
        point = Point(site['lon'], site['lat'])
        hits = sorted(codes[int(i)] for i in tree.query(point, predicate='intersects'))
        deepest, excluded = choose_deepest(hits, nodes)
        site['adminCandidates'] = [runtime_id(code, nodes) for code in deepest]
        site['admin'] = site['adminCandidates'][0] if len(deepest) == 1 else None
        if {k: v for k, v in before.items() if k not in ('admin', 'adminCandidates')} != {k: v for k, v in site.items() if k not in ('admin', 'adminCandidates')}:
            raise ValueError('Research site values changed')
        audit.append({'id': site['id'], 'name': site['name'], 'lon': site['lon'], 'lat': site['lat'],
                      'before': before.get('admin'), 'after': site['admin'], 'rawHits': hits,
                      'excludedHitAncestors': excluded, 'deepestCandidates': deepest,
                      'changed': before.get('admin') != site['admin'],
                      'candidatesChanged': before.get('adminCandidates', []) != site['adminCandidates']})
    return sites, audit


def row_identity(row, nodes):
    """Exact code/name + real ancestor context; no name-only or nearest fallback."""
    values = row['values']
    details = {'values': values[:6]}
    code_text, name = values[4].strip(), values[5].strip()
    if not re.fullmatch(r'\d{6}', code_text):
        return None, 'missing-explicit-target-code', details
    code = int(code_text)
    target = nodes.get(code)
    if not target:
        return None, 'target-code-not-in-official-tree', details
    if target['name'] != name:
        return None, 'same-code-name-changed-or-different', details | {'officialName': target['name']}
    chain = ancestry(code, nodes)
    province = next((nodes[c] for c in chain if nodes[c]['level'] == 'province'), None)
    if not province or values[1].strip() != province['name']:
        return None, 'province-name-ancestor-conflict', details | {'officialProvince': province}
    province_field = values[0].strip()
    if re.fullmatch(r'\d{6}', province_field) and int(province_field) in nodes:
        field_chain = ancestry(int(province_field), nodes)
        if province['adcode'] not in field_chain:
            return None, 'province-code-ancestor-conflict', details
        if nodes[int(province_field)]['level'] != 'province':
            details['provinceCodeFieldIsNotProvince'] = int(province_field)
    # Only real intermediate city ancestors constrain the old city columns.
    city_ancestors = [nodes[c] for c in chain[1:] if nodes[c]['level'] == 'city']
    if city_ancestors:
        city = city_ancestors[0]
        if values[2].strip() != str(city['adcode']) or values[3].strip() != city['name']:
            return None, 'city-code-or-name-ancestor-conflict', details | {'officialCityAncestor': city}
    elif values[2].strip() or values[3].strip():
        details['noRealIntermediateCity'] = {'originalCityCode': values[2], 'originalCityName': values[3], 'actualParent': target['parent']}
    return code, 'same-code-name-real-ancestors', details


def name_fallback(row, nodes, terminal_names, chains):
    values = row['values'];name = values[5].strip();province_name = values[1].strip();city_name = values[3].strip()
    candidates = []
    direct_context = {'', '市辖区', '县', '省直辖县级行政区划', '省直辖行政单位', '自治区直辖县级行政区划'}
    for code in terminal_names.get(name, []):
        chain = chains[code]
        province = next(nodes[c] for c in chain if nodes[c]['level'] == 'province')
        if province['name'] != province_name:
            continue
        cities = [nodes[c] for c in chain[1:] if nodes[c]['level'] == 'city']
        if cities and cities[0]['name'] != city_name:
            continue
        if not cities and city_name not in direct_context | {province['name'], nodes[code]['name']}:
            continue
        candidates.append(code)
    explicit = values[4].strip()
    if re.fullmatch(r'\d{6}', explicit) and int(explicit) in nodes:
        current = nodes[int(explicit)]
        chain = chains[int(explicit)]
        current_province = next((nodes[c] for c in chain if nodes[c]['level'] == 'province'), None)
        current_cities = [nodes[c]['name'] for c in chain[1:] if nodes[c]['level'] == 'city']
        # A known same-context code with a changed name remains a historical-name
        # review. Never reinterpret that rename as a different same-name entity.
        if current['name'] != name and current_province and current_province['name'] == province_name and (not current_cities or city_name in current_cities):
            return [], 'known-code-name-change-pending'
    return sorted(candidates), 'unique' if len(candidates) == 1 else 'ambiguous' if candidates else 'no-exact-ancestor-name'


def associate_classification(original, nodes, old_index):
    children = Counter(n['parent'] for n in nodes.values())
    chains = {code: ancestry(code, nodes) for code in nodes}
    terminal_names = {}
    for code, node in nodes.items():
        if node['level'] == 'district' or node['level'] in ('city', 'province') and not children[code]:
            terminal_names.setdefault(node['name'], []).append(code)
    result, audits = copy.deepcopy(original), []
    for before_sheet, sheet in zip(original['sheets'], result['sheets']):
        for before, row in zip(before_sheet['rows'], sheet['rows']):
            code, reason, details = row_identity(row, nodes)
            flags = []
            if code is None:
                primary_failure = reason
                options, outcome = name_fallback(row, nodes, terminal_names, chains)
                details.update(primaryFailureReason=primary_failure, nameFallbackCandidates=options, nameFallbackOutcome=outcome)
                if len(options) == 1:
                    code, reason = options[0], 'exact-ancestor-name-fallback'
                    explicit = row['values'][4].strip()
                    if re.fullmatch(r'\d{6}', explicit) and int(explicit) != code:
                        flags.append('explicit-code-name-conflict')
                        details['explicitCodeConflict'] = {'originalCode': explicit, 'identityAtOriginalCode': nodes.get(int(explicit)), 'selectedCode': code}
                    elif primary_failure == 'city-code-or-name-ancestor-conflict':
                        flags.append('explicit-ancestor-code-conflict')
                    details['selectedNameEvidence'] = {'name': nodes[code]['name'], 'ancestorChain': [nodes[c] for c in chains[code]], 'exactScopeCandidateCount': 1}
            row['admin'] = runtime_id(code, nodes) if code is not None else None
            row['match'] = 'datav-' + reason + (':' + ','.join(flags) if flags else '')
            previous = old_index.get(before.get('admin'))
            if previous and row['admin'] and previous['level'] != nodes[code]['level']:
                details['oldLayerChanged'] = {'id': previous['id'], 'level': previous['level'], 'newLevel': nodes[code]['level'], 'name': previous['name']}
            if {k: v for k, v in row.items() if k not in ('admin', 'match')} != {k: v for k, v in before.items() if k not in ('admin', 'match')}:
                raise ValueError('Original classification values changed')
            audits.append({'id': row['id'], 'sheet': sheet['id'], 'before': before.get('admin'), 'after': row['admin'],
                           'beforeMatch': before.get('match'), 'afterMatch': row['match'], 'reason': reason,
                           'changed': before.get('admin') != row['admin'], 'matchChanged': before.get('match') != row['match'], 'flags': flags, **details})
    return result, audits


def review_supplements(path, expected_sha, approved_ids, nodes, decisions=None, decisions_sha=None):
    """Read only explicitly selected, hash-pinned review items; never choose islands."""
    from shapely.geometry import shape
    if path is None:
        if approved_ids:
            raise ValueError('Approved supplement IDs require the exact review manifest')
        return []
    if len(approved_ids) != len(set(approved_ids)):
        raise ValueError('Duplicate approved supplement ID')
    checked(path, expected_sha)
    if approved_ids and (decisions is None or not decisions_sha):
        raise ValueError('Each applied supplement requires a pinned decision manifest')
    approvals = {}
    if decisions is not None:
        checked(decisions, decisions_sha)
        decision_data = json.loads(decisions.read_bytes())
        if decision_data.get('schema') != 'ptb-island-supplement-decisions/1' or decision_data.get('reviewItemsSha256') != expected_sha:
            raise ValueError('Decision manifest does not approve this exact review')
        approvals = {r['id']: r for r in decision_data['items']}
    review = json.loads(path.read_bytes())
    if review.get('schema') != 'ptb-island-supplement-review/1':
        raise ValueError('Unsupported island review schema')
    by_id = {r['id']: r for r in review['candidates']}
    if len(by_id) != len(review['candidates']) or not set(approved_ids) <= by_id.keys():
        raise ValueError('Duplicate/unknown approved supplement ID')
    selected, source_cache = [], {}
    for identity in approved_ids:
        item = copy.deepcopy(by_id[identity])
        approval = approvals.get(identity)
        if not approval or approval.get('decision') != 'approve' or sorted(approval['targets']) != sorted(t['adcode'] for t in item['proposedDisplayTargets']):
            raise ValueError('Supplement targets are not explicitly approved')
        if not item.get('selection', {}).get('rule') or not item.get('source', {}).get('sha256'):
            raise ValueError('Supplement lacks explicit source/selection evidence')
        relative = Path(item['candidateFile'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Candidate geometry path escapes review directory')
        candidate = path.parent / relative
        checked(candidate, item['candidateFileSha256'])
        data = json.loads(candidate.read_bytes())
        if data.get('type') != 'FeatureCollection' or len(data['features']) != 1:
            raise ValueError('Each supplement must be exactly one reviewed feature')
        geometry = data['features'][0]['geometry']
        value = shape(geometry)
        if value.geom_type != 'Polygon' or not value.is_valid or value.is_empty or value.area <= 0:
            raise ValueError('Supplement is not a valid, positive-area Polygon')
        basis = item['coordinateSystem']['basis']
        if basis not in ('datav-native-inferred', 'EPSG:4326'):
            raise ValueError('Supplement coordinate basis must be explicit')
        source = Path(item['source']['path'])
        checked(source, item['source']['sha256'])
        if str(source) not in source_cache:
            source_cache[str(source)] = json.loads(source.read_bytes())
        feature = source_cache[str(source)]['features'][item['source']['featureIndex']]
        if feature.get('properties') != item['source']['sourceProperties']:
            raise ValueError('Supplement source lineage properties changed')
        if item['source']['geometryPath'] != f"features[{item['source']['featureIndex']}].geometry.coordinates[{item['source']['componentIndex']}]":
            raise ValueError('Supplement source component path is inconsistent')
        component = feature['geometry']['coordinates'][item['source']['componentIndex']]
        original_polygon = {'type': 'Polygon', 'coordinates': [component] if feature['geometry']['type'] == 'MultiLineString' else component}
        if geometry != original_polygon:
            raise ValueError('Supplement differs from the exact reviewed source component')
        for target in item['proposedDisplayTargets']:
            node = nodes.get(target['adcode'])
            if not node or (node['level'], node['name']) != (target['level'], target['name']):
                raise ValueError('Supplement target is not the exact official tree identity')
        item['geometry'] = geometry
        item['approval'] = approval
        item['secondarySourceId'] = 'nlsc-taiping' if basis == 'EPSG:4326' else 'datav-reviewed-' + identity
        selected.append(item)
    return selected


def collect_supplements(groups, nodes):
    """Keep independently approved review bundles and their identities separate."""
    selected, reviews, seen = [], [], set()
    for group in groups:
        items = review_supplements(group['review'], group['reviewSha256'], group['approvedIds'], nodes,
                                   group['decisions'], group['decisionsSha256'])
        identities = {item['id'] for item in items}
        if seen & identities:
            raise ValueError('Duplicate approved supplement ID across review bundles')
        seen.update(identities)
        if group['review'] is not None:
            reviews.append({'reviewSha256': group['reviewSha256'], 'decisionsSha256': group['decisionsSha256'],
                            'approvedIds': list(group['approvedIds'])})
        for item in items:
            item['reviewManifestSha256'] = group['reviewSha256']
            item['decisionManifestSha256'] = group['decisionsSha256']
        selected.extend(items)
    return selected, reviews


def boundary_identity(bindings):
    """Content-bound versions prevent a new authority replacing older candidates."""
    if set(bindings) != set(BOUNDARY_BINDINGS):
        raise ValueError('Boundary identity requires all eight geography file bindings')
    sha = digest(json.dumps(bindings, sort_keys=True, separators=(',', ':')).encode('utf8'))
    return 'datav-areas-v3-inferred-gcj-wgs84-' + sha[:16], sha


def add_supplements(geometries, items, basis, stage):
    from shapely import union_all
    from shapely.geometry import shape
    rows = []
    for item in items:
        if item['coordinateSystem']['basis'] != basis:
            continue
        addition = shape(item['geometry'])
        for target in item['proposedDisplayTargets']:
            code = target['adcode'];before = geometries[code]
            after = union_all([before, addition], grid_size=0)
            if not after.is_valid or after.is_empty or after.geom_type not in ('Polygon', 'MultiPolygon'):
                raise ValueError('Reviewed supplement union is invalid')
            geometries[code] = after
            rows.append({'reviewId': item['id'], 'target': target, 'stage': stage, 'datum': basis,
                         'source': item['source'], 'candidateFileSha256': item['candidateFileSha256'],
                         'selectionRule': item['selection']['rule'], 'before': statistics(before), 'after': statistics(after),
                         'addedAreaNativeUnitsSquared': after.area - before.area,
                         'sourceId': item['secondarySourceId'], 'meaning': item.get('administrativeMeaning'),
                         'displayRole': item.get('displayRole'), 'dryLandMask': item.get('dryLandMask'),
                         'reviewManifestSha256': item.get('reviewManifestSha256'),
                         'decisionManifestSha256': item.get('decisionManifestSha256')})
    return rows


def drawing_layers(features, nodes):
    children = Counter(n['parent'] for n in nodes.values())
    result = {level: [copy.deepcopy(f) for code, f in features.items() if nodes[code]['level'] == level] for level in ('province', 'city', 'district')}
    fallbacks = []
    for code, feature in features.items():
        if nodes[code]['level'] == 'province' and not any(n['parent'] == code and n['level'] == 'city' for n in nodes.values()):
            fallback = copy.deepcopy(feature)
            fallback['properties']['renderFallback'] = 'city'
            result['city'].append(fallback)
            fallbacks.append({'id': feature['id'], 'actualLevel': 'province', 'drawLayer': 'city'})
        if not children[code] and nodes[code]['level'] != 'district':
            fallback = copy.deepcopy(feature)
            fallback['properties']['renderFallback'] = 'district'
            result['district'].append(fallback)
            fallbacks.append({'id': feature['id'], 'actualLevel': nodes[code]['level'], 'drawLayer': 'district'})
    return result, fallbacks


def build_runtime(normalized, raw, research, output, normalized_sha, research_sha, review=None, review_sha=None, approved=(), decisions=None, decisions_sha=None,
                  additional_review=None, additional_review_sha=None, additional_approved=(), additional_decisions=None, additional_decisions_sha=None):
    import shapely
    from shapely import union_all
    from shapely.geometry import shape, mapping
    if output.exists() or output == normalized or normalized in output.parents or output == raw or raw in output.parents or output == research or research in output.parents:
        raise ValueError('Use a new isolated output outside all input directories')
    checked(normalized / 'manifest.json', normalized_sha)
    normalized_manifest = json.loads((normalized / 'manifest.json').read_bytes())
    for name, sha in normalized_manifest['files'].items():
        checked(normalized / name, sha)
    checked(research / 'data-manifest.json', research_sha)
    research_manifest = json.loads((research / 'data-manifest.json').read_bytes())
    if {r['path'] for r in research_manifest['files']} != set(DATA_FILES):
        raise ValueError('Frozen research runtime must contain exactly the 16 contract files')
    for row in research_manifest['files']:
        checked(research / row['path'], row['sha256'])
    raw_inventory = json.loads((raw / 'source-inventory.json').read_bytes())
    if raw_inventory['rawSourceSetSha256'] != normalized_manifest['rawSourceSetSha256']:
        raise ValueError('Normalized/raw source sets differ')
    for row in raw_inventory['files']:
        if row.get('status') == 200:
            checked(raw / row['name'], row['sha256'])
    nodes = tree_index(json.loads((normalized / 'tree-index.json').read_bytes())['nodes'])
    native, normalized_refs = {}, {}
    for level in ('province', 'city', 'district'):
        for feature in json.loads((normalized / (level + '.geojson')).read_bytes())['features']:
            code = feature['properties']['adcode']
            if code in native or nodes[code]['level'] != level:
                raise ValueError('Duplicate/mismatched real administrative identity')
            native[code] = shape(feature['geometry']);normalized_refs[code] = feature['ptbCandidateSource']
    if set(native) != set(nodes) - {100000}:
        raise ValueError('Normalized objects do not cover the official tree')
    supplements, supplement_reviews = collect_supplements([
        {'review': review, 'reviewSha256': review_sha, 'approvedIds': approved, 'decisions': decisions, 'decisionsSha256': decisions_sha},
        {'review': additional_review, 'reviewSha256': additional_review_sha, 'approvedIds': additional_approved,
         'decisions': additional_decisions, 'decisionsSha256': additional_decisions_sha}], nodes)
    all_approved = [item['id'] for item in supplements]
    supplement_audit = add_supplements(native, supplements, 'datav-native-inferred', 'before-single-inverse')
    adapter = CoordinateAdapter()
    geometries = {code: shape(adapter.geometry(mapping(geometry))) for code, geometry in native.items()}
    supplement_audit += add_supplements(geometries, supplements, 'EPSG:4326', 'after-inverse-WGS84-bypass')
    invalid = [code for code, geometry in geometries.items() if not geometry.is_valid or geometry.is_empty]
    if invalid:
        raise ValueError('Coordinate derivation produced invalid geometries: ' + repr(invalid))
    features, index = {}, []
    for code, geometry in sorted(geometries.items()):
        node = nodes[code];chain = ancestry(code, nodes)
        province = next(c for c in chain if nodes[c]['level'] == 'province')
        secondary = sorted({item['secondarySourceId'] for item in supplements if any(t['adcode'] == code for t in item['proposedDisplayTargets'])})
        props = {'id': runtime_id(code, nodes), 'code': str(code), 'name': node['name'], 'level': node['level'],
                 'parent': runtime_id(node['parent'], nodes), 'province': runtime_id(province, nodes),
                 'source': SOURCE, 'sourceId': 'M19-DATAV', 'sourcePageVersionTime': '2021.5',
                 'sourceLevel': node['level'], 'sourceParentCode': node['parent'],
                 'coordinateAdapter': 'inferred-GCJ-to-WGS84', 'sourceGeometry': normalized_refs[code]}
        if secondary:
            props.update(supplementalSources=secondary, secondarySources=secondary)
        display_supplements = [{'reviewId': item['id'], 'sourceId': item['secondarySourceId'],
                                'displayRole': item['displayRole'], 'dryLandMask': item.get('dryLandMask'),
                                'meaning': item.get('administrativeMeaning')} for item in supplements
                               if item.get('displayRole') and any(t['adcode'] == code for t in item['proposedDisplayTargets'])]
        if display_supplements:
            props['supplementalDisplayAreas'] = display_supplements
        features[code] = {'type': 'Feature', 'id': props['id'], 'properties': props, 'geometry': mapping(geometry)}
        point = geometry.representative_point()
        index.append(props | {'center': [point.x, point.y], 'extent': list(geometry.bounds)})
    layers, fallback_audit = drawing_layers(features, nodes)
    territory = []
    provinces = [geometries[c] for c in geometries if nodes[c]['level'] == 'province']
    land = union_all(provinces, grid_size=0)
    if not land.is_valid or land.is_empty or land.geom_type not in ('Polygon', 'MultiPolygon'):
        raise ValueError('National administrative-display union is invalid')
    province_codes = sorted(c for c in geometries if nodes[c]['level'] == 'province')
    land_overlaps = []
    for i, left in enumerate(province_codes):
        for right in province_codes[i+1:]:
            a, b = geometries[left], geometries[right]
            if a.intersects(b):
                overlap = a.intersection(b)
                if overlap.area > 0:
                    land_overlaps.append({'codes': [left, right], 'areaNativeUnitsSquared': overlap.area, 'bounds': list(overlap.bounds)})
    territory.append({'type': 'Feature', 'id': 'datav-administrative-display-area', 'properties': {
        'source': SOURCE, 'renderRole': 'administrative-display-area', 'administrativeEntity': False,
        'pointAssociationAllowed': False, 'dryLandMask': False,
        'areaMeaning': 'Union of source administrative display ranges; not physical dry-land or national land area'}, 'geometry': mapping(land)})
    boundary = json.loads((raw / '100000_boundary.json').read_bytes())
    for i, feature in enumerate(boundary['features']):
        kind = feature['properties']['type'];geometry = feature['geometry']
        if geometry['type'] != 'MultiLineString':
            raise ValueError('Unexpected official boundary geometry type')
        territory.append({'type': 'Feature', 'id': 'datav-boundary-' + str(i), 'properties': copy.deepcopy(feature['properties']) | {
            'source': SOURCE, 'renderRole': 'disputed-line' if kind == '争议' else 'boundary-line',
            'line': 'dash' if kind == '争议' else 'land', 'fill': False, 'boundaryType': kind,
            'sourceCrsDeclaration': boundary.get('crs'), 'coordinateAdapter': 'inferred-GCJ-to-WGS84',
            'administrativeEntity': False, 'pointAssociationAllowed': False}, 'geometry': adapter.geometry(geometry)})
    national = json.loads((raw / '100000_full.json').read_bytes())
    symbols = [f for f in national['features'] if f['properties'].get('adcode') == '100000_JD']
    if len(symbols) != 1 or symbols[0]['geometry']['type'] != 'MultiPolygon' or len(symbols[0]['geometry']['coordinates']) != 10:
        raise ValueError('Expected exact ten-part national boundary symbol')
    for i, polygon in enumerate(symbols[0]['geometry']['coordinates']):
        territory.append({'type': 'Feature', 'id': 'datav-jd-' + str(i), 'properties': {
            'source': SOURCE, 'renderRole': 'boundary-symbol', 'sourceSymbol': '100000_JD', 'sourcePart': i,
            'administrativeEntity': False, 'pointAssociationAllowed': False, 'landArea': False},
            'geometry': adapter.geometry({'type': 'Polygon', 'coordinates': polygon})})
    source_sites = json.loads((research / 'sites.json').read_bytes())
    sites, site_audit = associate_sites(source_sites, geometries, nodes)
    original_classification = json.loads((research / 'classification.json').read_bytes())
    old_index = {r['id']: r for r in json.loads((research / 'map-index.json').read_bytes())}
    classification, class_audit = associate_classification(original_classification, nodes, old_index)
    output.mkdir(parents=True)
    for name in (*IMMUTABLE, 'maps/world.json'):
        atomic(output / name, (research / name).read_bytes())
    for name, value in [('sites.json', sites), ('classification.json', classification), ('map-index.json', index), ('maps/territory.json', {'type': 'FeatureCollection', 'features': territory})]:
        atomic(output / name, compact(value))
    for level, values in layers.items():
        atomic(output / 'maps' / (level + '.json'), compact({'type': 'FeatureCollection', 'features': values}))
    associations = {'pointMethod': 'Unique deepest actual ancestor-chain hit; incomparable conflicts retained; no nearest matching',
                    'sites': site_audit, 'classification': class_audit,
                    'siteMatched': sum(bool(s['admin']) for s in sites), 'siteAmbiguous': sum(len(s['adminCandidates']) > 1 for s in sites),
                    'siteUnmatched': sum(not s['adminCandidates'] for s in sites), 'siteAdminChanges': sum(r['changed'] for r in site_audit),
                    'classificationMatched': sum(bool(r['after']) for r in class_audit),
                    'classificationAdminChanges': sum(r['changed'] for r in class_audit),
                    'classificationMatchChanges': sum(r['matchChanged'] for r in class_audit),
                    'classificationSameCode': sum(r['reason'] == 'same-code-name-real-ancestors' for r in class_audit),
                    'classificationExactAncestorNameFallback': sum(r['reason'] == 'exact-ancestor-name-fallback' for r in class_audit),
                    'classificationCodeConflictFallback': sum(bool(r['flags']) for r in class_audit),
                    'classificationPending': sum(not r['after'] for r in class_audit),
                    'classificationReasons': dict(Counter(r['reason'] for r in class_audit)),
                    'researchCoordinateChanges': 0, 'originalClassificationValueChanges': 0}
    atomic(output / 'association-audit.json', encoded(associations))
    derivation = {'coordinateAdapter': {'id': 'inferred-GCJ-to-WGS84', 'function': 'prepare_geolinguistics.gcj_inverse',
                   'functionSha256': digest(inspect.getsource(gcj_inverse).encode()), 'sourceFileSha256': digest((Path(__file__).parent / 'prepare_geolinguistics.py').read_bytes()),
                   'positionOccurrences': adapter.positions, 'uniquePositionsConvertedOnce': len(adapter.cache),
                   'status': 'Inferred display derivation; not officially confirmed', 'boundaryDeclarationConflict': boundary.get('crs'),
                   'WGS84SupplementsBypassInverse': [s['id'] for s in supplements if s['coordinateSystem']['basis'] == 'EPSG:4326']},
                  'normalizedManifestSha256': normalized_sha, 'rawSourceSetSha256': normalized_manifest['rawSourceSetSha256'],
                  'supplementReviewSha256': review_sha, 'decisionManifestSha256': decisions_sha,
                  'supplementReviews': supplement_reviews, 'approvedSupplementIds': all_approved, 'supplements': supplement_audit,
                  'fallbacks': fallback_audit, 'realObjectsBySourceLevel': dict(Counter(n['level'] for c, n in nodes.items() if c != 100000)),
                  'drawingFeatureCounts': {k: len(v) for k, v in layers.items()},
                  'administrativeDisplayUnion': {'inputProvinceCount': len(provinces), 'sumInputAreasNativeSquared': sum(g.area for g in provinces),
                                    'result': statistics(land), 'sumInputAreaMinusUnionArea': sum(g.area for g in provinces) - land.area,
                                    'inputProvinces': [{'code': code, 'statistics': statistics(geometries[code])} for code in province_codes],
                                    'positiveAreaOverlapPairs': land_overlaps,
                                    'administrativeEntity': False, 'pointAssociationAllowed': False, 'symbolsExcluded': True,
                                    'dryLandMask': False, 'nationalLandAreaComputationAllowed': False,
                                    'interpretation': 'Source administrative display range only; retains submerged-bank display polygons. Native angular area statistics are geometry diagnostics.'},
                  'territoryRoles': dict(Counter(f['properties']['renderRole'] for f in territory)),
                  'researchFilesCopiedByteForByte': {name: digest((research / name).read_bytes()) for name in IMMUTABLE},
                  'sourceNamesPreserved': True, 'rawCoordinatesUnchanged': True}
    atomic(output / 'derivation-audit.json', encoded(derivation))
    manifest = data_manifest(output)
    bindings = {r['path']: {'bytes': r['bytes'], 'sha256': r['sha256']} for r in manifest['files'] if r['path'] in BOUNDARY_BINDINGS}
    boundary_version, binding_sha = boundary_identity(bindings)
    manifest['distribution'] = 'Isolated GeoAtlas learning/exchange runtime candidate; no public activation'
    manifest['boundaries'] = {'schema': 'ptb-geolinguistics-boundary-provenance/1', 'version': boundary_version,
        'status': 'isolated-candidate-review-required', 'coordinateSystem': 'EPSG:4326 display derived via inferred-GCJ-to-WGS84',
        'coordinateInferenceOfficiallyConfirmed': False, 'originalBoundaryCrsDeclaration': boundary.get('crs'),
        'bindings': bindings, 'bindingsSha256': binding_sha, 'normalizedManifestSha256': normalized_sha, 'rawSourceSetSha256': normalized_manifest['rawSourceSetSha256'],
        'sources': [{'id': SOURCE, 'sourceId': 'M19-DATAV', 'attribution': 'DataV GeoAtlas / 高德开放平台',
                     'sourceUrl': 'https://datav.aliyun.com/portal/school/atlas/area_selector', 'sourceVersion': 'areas_v3', 'pageVersionTime': '2021.5',
                     'license': '仅供学习和交流使用', 'licenseUrl': 'https://help.aliyun.com/zh/datav/datav-7-0/user-guide/datav-geoatlas-widgets/',
                     'dataIsProjectMIT': False}, {'id': 'natural-earth-background', 'sourceUrl': 'https://www.naturalearthdata.com/',
                     'license': 'Public domain', 'runtimePath': 'maps/world.json', 'sha256': digest((output / 'maps/world.json').read_bytes())}],
        'supplementReviewSha256': review_sha, 'decisionManifestSha256': decisions_sha, 'supplementReviews': supplement_reviews, 'approvedSupplementIds': all_approved,
        'supplementSources': [{'id': s['secondarySourceId'], 'reviewId': s['id'], 'sourceSha256': s['source']['sha256'],
                               'datum': s['coordinateSystem']['basis'], 'license': s.get('license'), 'targetCodes': [t['adcode'] for t in s['proposedDisplayTargets']],
                               'displayRole': s.get('displayRole'), 'dryLandMask': s.get('dryLandMask'), 'meaning': s.get('administrativeMeaning'),
                               'reviewManifestSha256': s['reviewManifestSha256'], 'decisionManifestSha256': s['decisionManifestSha256']} for s in supplements],
        'audit': {name: digest((output / name).read_bytes()) for name in ('association-audit.json', 'derivation-audit.json')}}
    atomic(output / 'data-manifest.json', encoded(manifest))
    # Validate all frozen inputs again and prove every immutable output byte.
    for row in research_manifest['files']:
        checked(research / row['path'], row['sha256'])
    for name in IMMUTABLE:
        if (output / name).read_bytes() != (research / name).read_bytes():
            raise ValueError('Immutable research output changed')
    for row in raw_inventory['files']:
        if row.get('status') == 200:
            checked(raw / row['name'], row['sha256'])
    differences = [{'path': name, 'beforeBytes': (research / name).stat().st_size, 'afterBytes': (output / name).stat().st_size,
                    'beforeSha256': digest((research / name).read_bytes()), 'afterSha256': digest((output / name).read_bytes()),
                    'changed': (research / name).read_bytes() != (output / name).read_bytes()} for name in DATA_FILES]
    report = {'status': 'isolated-runtime-candidate-review-required', 'runtimeFiles': 16, 'runtimeBytes': sum(r['bytes'] for r in manifest['files']),
              'producerSha256': digest(Path(__file__).read_bytes()), 'provenanceVersion': boundary_version, 'bindingsSha256': binding_sha,
              'manifestSha256': digest((output / 'data-manifest.json').read_bytes()), 'realAdministrativeObjects': len(index),
              'levels': derivation['realObjectsBySourceLevel'], 'fallbackCount': len(fallback_audit), 'drawingCounts': derivation['drawingFeatureCounts'],
              'associations': {k: v for k, v in associations.items() if k not in ('sites', 'classification')},
              'differences': differences, 'rawInputUnchanged': True, 'researchSourceUnchanged': True,
              'researchManifestSha256': research_sha, 'normalizedManifestSha256': normalized_sha,
              'approvedSupplements': all_approved, 'sourceFieldExample': next(r for r in index if r['code'] == '460000'),
              'limitations': ['Coordinate adapter is inferred, with original CRS84 conflict preserved.', 'Source page version2021.5 is not certified2026 administrative currency.',
                              'Validity/code coverage is not proof of complete legal boundaries; no nearest or arbitrary conflict resolution.']}
    atomic(output / 'runtime-build-report.json', encoded(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normalized', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--research', type=Path, required=True, help='Exact frozen-candidate-5/runtime directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-normalized-sha256', required=True)
    parser.add_argument('--expected-research-sha256', required=True)
    parser.add_argument('--supplement-review', type=Path)
    parser.add_argument('--expected-supplement-sha256')
    parser.add_argument('--approved-supplement', action='append', default=[])
    parser.add_argument('--decisions', type=Path)
    parser.add_argument('--expected-decisions-sha256')
    parser.add_argument('--additional-supplement-review', type=Path)
    parser.add_argument('--expected-additional-supplement-sha256')
    parser.add_argument('--approved-additional-supplement', action='append', default=[])
    parser.add_argument('--additional-decisions', type=Path)
    parser.add_argument('--expected-additional-decisions-sha256')
    parser.add_argument('--shapely-path', type=Path, required=True)
    args = parser.parse_args()
    for directory in (args.normalized.absolute(), args.raw.absolute(), args.research.absolute(), args.output.absolute()):
        for path in (directory, *directory.parents):
            if path.is_symlink() or path.exists() and getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
                parser.error('Linked input/output paths are not allowed')
    if not args.output.resolve().is_relative_to(args.normalized.resolve().parent):
        parser.error('Runtime output must stay in this isolated DataV candidate directory')
    sys.path.insert(0, str(args.shapely_path.resolve()))
    result = build_runtime(args.normalized.resolve(), args.raw.resolve(), args.research.resolve(), args.output.resolve(),
                           args.expected_normalized_sha256, args.expected_research_sha256,
                           args.supplement_review, args.expected_supplement_sha256, args.approved_supplement,
                           args.decisions, args.expected_decisions_sha256,
                           args.additional_supplement_review, args.expected_additional_supplement_sha256,
                           args.approved_additional_supplement, args.additional_decisions, args.expected_additional_decisions_sha256)
    print(json.dumps({k: v for k, v in result.items() if k != 'differences'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
