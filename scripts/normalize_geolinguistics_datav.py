"""Reviewable, source-native GeoAtlas geometry candidate; never writes raw/runtime.

Invalid Polygon members are repaired independently with structure/keep_collapsed,
then their area components are unioned within that original Feature. Symbols,
boundary lines and collapsed dimensions remain separately auditable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys

from prepare_geolinguistics_datav import atomic, digest, encoded, file_features, tree_index


def compact(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf8')


def coordinates(geometry):
    """All distinct original vertices, with no tolerance, rounding or reprojection."""
    def visit(value):
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (int, float)):
            yield tuple(value)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from visit(child)
    if geometry.get('type') == 'GeometryCollection':
        return set().union(*(coordinates(g) for g in geometry['geometries'])) if geometry['geometries'] else set()
    return set(visit(geometry.get('coordinates', [])))


def components(value):
    if hasattr(value, 'geoms'):
        for child in value.geoms:
            yield from components(child)
    elif not value.is_empty:
        yield value


def statistics(value):
    from shapely import get_num_coordinates
    from shapely.validation import explain_validity
    parts = list(components(value))
    polygons = [p for p in parts if p.geom_type == 'Polygon']
    return {'type': value.geom_type, 'valid': bool(value.is_valid), 'validity': explain_validity(value),
            'empty': bool(value.is_empty), 'areaNativeUnitsSquared': float(value.area),
            'bounds': list(value.bounds) if not value.is_empty else [], 'polygonParts': len(polygons),
            'holes': sum(len(p.interiors) for p in polygons),
            'coordinateVerticesIncludingRingClosures': int(get_num_coordinates(value)),
            'componentTypes': dict(Counter(p.geom_type for p in parts))}


def introduced_vertices(raw_geometry, result_geometry):
    """Record every newly noded vertex plus original segment support evidence."""
    from shapely import STRtree
    from shapely.geometry import LineString, Point
    old, new = coordinates(raw_geometry), coordinates(result_geometry)
    polygon_members = [raw_geometry['coordinates']] if raw_geometry['type'] == 'Polygon' else raw_geometry['coordinates']
    segments, paths = [], []
    for polygon_index, polygon in enumerate(polygon_members):
        for ring_index, ring in enumerate(polygon):
            for segment_index, (a, b) in enumerate(zip(ring, ring[1:])):
                if a == b:
                    continue
                segments.append(LineString([a, b]))
                paths.append([polygon_index, ring_index, segment_index])
    index = STRtree(segments) if segments else None
    tolerance = 1e-10  # Native angular units; diagnostic support search only.
    added = []
    for vertex in sorted(new - old):
        point = Point(vertex)
        nearby = index.query(point, predicate='dwithin', distance=tolerance) if index is not None else []
        supports = [{'path': paths[int(i)], 'distanceNativeUnits': float(segments[int(i)].distance(point))} for i in nearby]
        supports.sort(key=lambda r: r['path'])
        added.append({'coordinate': list(vertex), 'supportingOriginalSegments': supports,
                      'intersectionSupported': len(supports) >= 2})
    return {'addedVertices': added, 'removedVertices': [list(v) for v in sorted(old - new)],
            'supportSearchToleranceNativeUnits': tolerance,
            'interpretation': 'Exact coordinate-set difference. All new values are retained; support tolerance diagnoses source segments and never changes coordinates.'}


def normalize_geometry(raw_geometry):
    from shapely import make_valid, union_all
    from shapely.geometry import shape, mapping
    original = shape(raw_geometry)
    before = statistics(original)
    if original.geom_type not in ('Polygon', 'MultiPolygon'):
        raise ValueError('Administrative geometry must be Polygon/MultiPolygon')
    if original.is_valid and not original.is_empty:
        return copy.deepcopy(raw_geometry), {'repaired': False, 'before': before, 'after': before}, []
    members = [raw_geometry['coordinates']] if raw_geometry['type'] == 'Polygon' else raw_geometry['coordinates']
    area_parts, collapsed, member_audit = [], [], []
    for index, member in enumerate(members):
        polygon = shape({'type': 'Polygon', 'coordinates': member})
        repaired = make_valid(polygon, method='structure', keep_collapsed=True)
        row = {'polygonIndex': index, 'before': statistics(polygon), 'after': statistics(repaired)}
        if not polygon.is_valid or repaired.geom_type != 'Polygon':
            row['resultGeometry'] = mapping(repaired)
        for part in components(repaired):
            if part.geom_type == 'Polygon' and part.area > 0:
                area_parts.append(part)
            else:
                collapsed.append({'polygonIndex': index, 'geometry': mapping(part), 'statistics': statistics(part)})
        member_audit.append(row)
    result = union_all(area_parts, grid_size=0) if area_parts else None
    if result is not None and (not result.is_valid or result.geom_type not in ('Polygon', 'MultiPolygon')):
        raise ValueError('Per-feature union did not produce valid area geometry')
    mapped = mapping(result) if result is not None and not result.is_empty else None
    added = introduced_vertices(raw_geometry, mapped) if mapped else {
        'addedVertices': [], 'removedVertices': [list(v) for v in sorted(coordinates(raw_geometry))]}
    audit = {'repaired': True, 'before': before, 'after': statistics(result) if result is not None else None,
             'method': 'Per-Polygon make_valid(method=structure,keep_collapsed=True), then within-Feature union_all(grid_size=0)',
             'polygonMembers': member_audit, 'collapsedComponents': collapsed,
             'sumRepairedPolygonAreasNativeUnitsSquared': sum(p.area for p in area_parts),
             'sumRepairedAreaMinusUnionAreaNativeUnitsSquared': sum(p.area for p in area_parts) - (result.area if result is not None else 0),
             'areaDeltaNativeUnitsSquared': (result.area if result is not None else 0) - original.area,
             **added}
    return mapped, audit, collapsed


def special_tree(index):
    children = defaultdict(list)
    for node in index.values():
        children[node['parent']].append(node['adcode'])
    direct = [copy.deepcopy(n) for n in index.values() if n['level'] == 'district' and index[n['parent']]['level'] == 'province']
    # This is a structural code-pattern report, not a new administrative level.
    source_city_leaf_codes = [copy.deepcopy(n) for n in index.values() if n['level'] == 'city' and n['adcode'] % 100 != 0 and not children[n['adcode']]]
    city_leaves = [copy.deepcopy(n) for n in index.values() if n['level'] == 'city' and not children[n['adcode']]]
    return {'districtsDirectlyUnderProvince': direct,
            'districtDirectParentCounts': dict(Counter(str(n['parent']) for n in direct)),
            'provinceDirectLeafEntitiesSourceLabelsCityNon00Code': source_city_leaf_codes,
            'sourceCityLeaves': city_leaves,
            'taiwan': {'node': index.get(710000), 'children': children.get(710000, []),
                       'sourceScope': 'Province only in this exact official tree; no invented city/district nodes'},
            'interpretation': 'Keep official source level/parent unchanged. City is an upstream display layer and must not be relabelled as prefecture level. No synthetic intermediate nodes.'}, dict(children)


def select_sources(source, index):
    candidates, auxiliary = defaultdict(list), []
    receipts = {r['name']: r for r in json.loads((source / 'source-inventory.json').read_bytes())['files'] if r.get('status') == 200}
    for name, data in file_features(source):
        for position, feature in enumerate(data['features']):
            code = (feature.get('properties') or {}).get('adcode')
            reference = {'file': name, 'rawSha256': receipts[name]['sha256'], 'feature': position}
            if isinstance(code, int) and code in index and index[code]['level'] != 'country':
                if feature['geometry']['type'] not in ('Polygon', 'MultiPolygon'):
                    raise ValueError('Non-area administrative input: ' + str(code))
                candidates[code].append((reference, feature))
            else:
                auxiliary.append((reference, feature))
    selected = {}
    for code, node in sorted(index.items()):
        if node['level'] == 'country':
            continue
        options = candidates.get(code, [])
        if not options:
            raise ValueError('Missing official tree geometry: ' + str(code))
        def rank(option):
            name = option[0]['file']
            if node['level'] == 'province':
                return (0 if name == '100000_full.json' else 1, name)
            if node['level'] == 'city':
                return (0 if name == '100000_full_city.json' else 1, name)
            return (0 if name.endswith('_full_district.json') else 1 if name != '100000_full_city.json' else 2, name)
        options.sort(key=rank)
        selected[code] = (options[0][0], options[0][1], [r for r, _ in options])
    return selected, auxiliary


def auxiliary_audit(reference, feature):
    from shapely.geometry import shape, LineString
    geometry = feature['geometry']
    try:
        stats = statistics(shape(geometry))
    except Exception as error:
        stats = {'error': type(error).__name__ + ': ' + str(error)}
    degenerate = []
    if geometry['type'] == 'MultiLineString':
        for index, line in enumerate(geometry['coordinates']):
            try:
                value = LineString(line)
                detail = statistics(value)
                if value.is_valid and not value.is_empty and value.length > 0:
                    continue
            except Exception as error:
                detail = {'error': type(error).__name__ + ': ' + str(error)}
            degenerate.append({'component': index, 'originalCoordinates': copy.deepcopy(line), 'statistics': detail})
    return {'source': reference, 'properties': copy.deepcopy(feature.get('properties', {})), 'statistics': stats,
            'degenerateLineComponents': degenerate, 'administrativeEntity': False,
            'pointAssociationAllowed': False, 'unchanged': True}


def build_candidate(source, output, expected_input_sha):
    import shapely
    from shapely.geometry import shape
    if output.exists():
        raise ValueError('Use a new independent output directory; existing candidate is never replaced')
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Output must be separate from raw input and its ancestors')
    inventory = json.loads((source / 'source-inventory.json').read_bytes())
    records = [r for r in inventory['files'] if r.get('status') == 200]
    raw_set = {}
    for row in records:
        path = source / row['name']
        if path.is_symlink() or digest(path.read_bytes()) != row['sha256']:
            raise ValueError('Raw input SHA mismatch: ' + row['name'])
        raw_set[row['name']] = row['sha256']
    input_sha = digest(encoded(dict(sorted(raw_set.items()))))
    if input_sha != expected_input_sha or input_sha != inventory['rawSourceSetSha256']:
        raise ValueError('Frozen raw source set identity mismatch')
    index = tree_index(json.loads((source / 'all.json').read_bytes()))
    selected, auxiliary = select_sources(source, index)
    output.mkdir(parents=True)
    layers, audits, collapsed_features, source_index, unavailable = defaultdict(list), [], [], {}, []
    for code, (reference, feature, alternatives) in selected.items():
        geometry, audit, collapsed = normalize_geometry(feature['geometry'])
        node = index[code]
        source_index[str(code)] = {'selected': reference, 'availableRawOccurrences': alternatives,
                                  'areaAvailable': geometry is not None}
        if geometry is None:
            unavailable.append(code)
        else:
            if not shape(geometry).is_valid:
                raise ValueError('Invalid normalized geometry: ' + str(code))
            result = copy.deepcopy(feature)
            result.update(id=node['level'] + ':' + str(code), geometry=geometry, ptbCandidateSource=reference)
            layers[node['level']].append(result)
        if audit['repaired']:
            audit.update(adcode=code, name=node.get('name'), level=node['level'], parent=node['parent'], source=reference)
            audits.append(audit)
        for part in collapsed:
            collapsed_features.append({'type': 'Feature', 'geometry': part['geometry'],
                                       'properties': {'sourceAdcode': code, 'sourceFile': reference['file'],
                                                      'sourceFeature': reference['feature'], 'sourcePolygon': part['polygonIndex'],
                                                      'role': 'collapsed-repair-component', 'administrativeEntity': False}})
    auxiliary_records = [auxiliary_audit(reference, feature) for reference, feature in auxiliary]
    auxiliary_features = [copy.deepcopy(feature) | {'ptbCandidateSource': reference} for reference, feature in auxiliary]
    for level in ('province', 'city', 'district'):
        atomic(output / (level + '.geojson'), compact({'type': 'FeatureCollection', 'features': layers[level]}))
    atomic(output / 'auxiliary-symbols-lines.raw.geojson', compact({'type': 'FeatureCollection', 'features': auxiliary_features}))
    atomic(output / 'collapsed-repair-components.geojson', compact({'type': 'FeatureCollection', 'features': collapsed_features}))
    specials, children = special_tree(index)
    tree = {'nodes': list(index.values()), 'children': children, 'features': source_index, 'specialStructures': specials}
    atomic(output / 'tree-index.json', encoded(tree))
    repair = {'rawSourceSetSha256': input_sha, 'coordinatePolicy': 'Untransformed source-native coordinates; CRS certification/adapter pending independent review',
              'algorithm': 'Only invalid administrative features: per Polygon structure/keep_collapsed=True, then within-Feature area union; no whole MultiPolygon linework',
              'areaUnit': 'Squared native angular coordinate units, not physical/geodesic area; raw invalid areas may double-count overlap',
              'precisionPolicy': 'No rounding, simplification, snapping or set_precision; union grid_size=0',
              'software': {'shapely': shapely.__version__, 'geos': shapely.geos_version_string},
              'repairedFeatures': audits, 'collapsedComponents': len(collapsed_features),
              'areaUnavailableCodes': unavailable, 'auxiliary': auxiliary_records,
              'notPerformed': ['island supplementation', 'cross-feature/cross-layer union', 'source mixing', 'administrative reassignment', 'research point association', 'coordinate conversion']}
    atomic(output / 'repair-audit.json', encoded(repair))
    # Recheck every raw byte after all calculations and writes.
    for name, sha in raw_set.items():
        if digest((source / name).read_bytes()) != sha:
            raise ValueError('Raw input changed during build: ' + name)
    file_shas = {p.name: digest(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}
    manifest = {'kind': 'reviewable-native-geometry-candidate', 'status': 'review_required',
                'version': 'areas_v3', 'sourcePageVersionTime': '2021.5', 'rawSourceSetSha256': input_sha,
                'files': file_shas, 'counts': {level: len(layers[level]) for level in ('province', 'city', 'district')},
                'repairedAdministrativeFeatures': len(audits), 'collapsedComponents': len(collapsed_features),
                'areaUnavailableCodes': unavailable, 'rawUnchanged': True,
                'dataUseNotice': 'DataV GeoAtlas / 高德开放平台，学习和交流使用；数据不采用项目 MIT 许可',
                'limitations': ['Geometry validity does not certify territorial/island completeness or coordinate registration.',
                                'No runtime/application/current data changed. No island additions or research associations.']}
    atomic(output / 'manifest.json', encoded(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-input-sha256', required=True)
    parser.add_argument('--shapely-path', type=Path, required=True)
    args = parser.parse_args()
    for directory in (args.input.absolute(), args.output.absolute()):
        for path in (directory, *directory.parents):
            if path.is_symlink() or path.exists() and getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
                parser.error('Linked input/output paths are not allowed')
    sys.path.insert(0, str(args.shapely_path.resolve()))
    result = build_candidate(args.input.resolve(), args.output.resolve(), args.expected_input_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
