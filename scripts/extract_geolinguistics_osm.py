"""Extract administrative polygons from a verified fixed PBF into an audit stage.

Requires task-local pyosmium and shapely, supplied with --tools (repeatable).
Keeps source relations and exact unsimplified polygon coordinates. No point
association, name fuzzy matching, CRS offset or topology repair is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def extract(source, output, relation_ids=None):
    import osmium
    from shapely.geometry import shape

    receipt_path = source.with_suffix(source.suffix + '.receipt.json')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('status') != 'verified-download' or digest(source) != receipt['sha256']:
        raise ValueError('Source PBF does not match its successful frozen download receipt')
    if output.exists():
        raise ValueError('Candidate output already exists; use a new isolated directory')
    output.mkdir(parents=True)
    scratch = output / 'scratch'
    scratch.mkdir()
    tempfile.tempdir = str(scratch)
    subset = output / 'administrative.osm.pbf'
    relations, nested = {}, []
    reader = osmium.io.Reader(str(source), osmium.osm.NOTHING)
    header = reader.header()
    timestamp = header.get('osmosis_replication_timestamp')
    reader.close()
    if timestamp != receipt['snapshot']:
        raise ValueError(f'PBF timestamp mismatch: {timestamp!r} != {receipt["snapshot"]!r}')
    print('Reading relation metadata', flush=True)
    processor = osmium.FileProcessor(str(source), osmium.osm.RELATION).with_filter(osmium.filter.TagFilter(('boundary', 'administrative')))
    with osmium.BackReferenceWriter(str(subset), ref_src=str(source), relation_depth=3) as writer:
        for relation in processor:
            tags = dict(relation.tags)
            if (relation_ids is not None and relation.id not in relation_ids) or (relation_ids is None and tags.get('admin_level') not in {'3', '4', '5', '6'}):
                continue
            members = [{'type': m.type, 'ref': m.ref, 'role': m.role} for m in relation.members]
            relations[relation.id] = {
                'id': relation.id, 'version': relation.version, 'timestamp': str(relation.timestamp),
                'tags': tags, 'members': members,
            }
            # Label/admin-centre/subarea members do not contribute to a boundary
            # polygon. Retain them in the metadata above, omit from geometry-only
            # extraction to avoid recursively copying all township data.
            geometry_members = [(m['type'], m['ref'], m['role']) for m in members
                                if m['type'] in {'w', 'r'} and m['role'] in {'outer', 'inner', ''}]
            if any(m[0] == 'r' for m in geometry_members):
                nested.append(relation.id)
            writer.add(relation.replace(members=geometry_members))
    save(output / 'relations.json', {'snapshot': timestamp, 'relations': list(relations.values()),
         'source': receipt, 'nestedGeometryRelations': nested})
    print(f'Assembling {len(relations)} administrative relations', flush=True)
    factory = osmium.geom.GeoJSONFactory()
    seen, invalid, features = set(), [], []
    for area in osmium.FileProcessor(str(subset)).with_locations('flex_mem').with_areas():
        if not area.is_area() or area.from_way() or area.orig_id() not in relations:
            continue
        rid = area.orig_id()
        seen.add(rid)
        try:
            geometry = json.loads(factory.create_multipolygon(area))
            polygon = shape(geometry)
            if polygon.is_empty or not polygon.is_valid or polygon.geom_type not in {'Polygon', 'MultiPolygon'}:
                raise ValueError('Empty, invalid or non-polygon geometry')
            relation = relations[rid]
            features.append({'type': 'Feature', 'id': f'osm:relation:{rid}',
                'properties': {'osmId': rid, 'osmVersion': relation['version'], 'snapshot': timestamp,
                               'tags': relation['tags']}, 'geometry': geometry})
        except Exception as error:
            invalid.append({'osmId': rid, 'reason': str(error)})
    save(output / 'boundaries.geojson', {'type': 'FeatureCollection', 'features': features})
    summary = {'source': receipt, 'snapshot': timestamp, 'relations': len(relations),
               'levels': dict(Counter(r['tags']['admin_level'] for r in relations.values())),
               'validPolygons': len(features), 'invalidPolygons': invalid,
               'unassembled': [{'osmId': rid, 'name': r['tags'].get('name'),
                               'admin_level': r['tags']['admin_level']} for rid, r in relations.items() if rid not in seen],
               'nestedGeometryRelations': nested, 'coordinateSystem': 'WGS84 (OSM); unchanged',
               'files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)}
                         for p in [subset, output / 'relations.json', output / 'boundaries.geojson']]}
    save(output / 'extraction-audit.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {'source', 'files', 'unassembled'}}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tools', type=Path, action='append', default=[])
    parser.add_argument('--relation-ids', type=int, nargs='+', help='An explicit reviewed additional administrative relation set')
    args = parser.parse_args()
    for path in args.tools:
        sys.path.insert(0, str(path.resolve()))
    output = args.output.resolve()
    active = Path(__file__).resolve().parents[1] / 'local-data'
    if output == active or active in output.parents:
        parser.error('Output must be an isolated audit directory outside local-data')
    extract(args.input.resolve(), output, set(args.relation_ids) if args.relation_ids else None)


if __name__ == '__main__':
    main()
