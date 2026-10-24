"""Audit and complete clipped administrative members at the same OSM snapshot.

Only missing ways of exact official-name/level relations are requested. Existing
PBF objects are never replaced. Each request, response hash and member list is
retained in an isolated directory. No topology repair or coordinate changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlencode


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    part.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract', type=Path, required=True)
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tools', type=Path, action='append', default=[])
    parser.add_argument('--stage', choices=['audit', 'fetch', 'fetch-api', 'assemble', 'polygonize', 'supplement-relations'], required=True)
    parser.add_argument('--additional-relation', type=Path, action='append', default=[])
    parser.add_argument('--endpoint', default='https://overpass-api.de/api/interpreter')
    parser.add_argument('--batch-size', type=int, default=40)
    args = parser.parse_args()
    for tool in args.tools:
        sys.path.insert(0, str(tool.resolve()))
    import osmium
    from prepare_geolinguistics_open_maps import names
    from fetch_geolinguistics_open_maps import acquire
    output, extract = args.output.resolve(), args.extract.resolve()
    if output == extract or extract in output.parents or 'local-data' in output.parts:
        parser.error('Use an independent isolated completion directory')
    output.mkdir(parents=True, exist_ok=True)
    metadata = load(extract / 'relations.json')
    official_names = {(str({1: 4, 2: 5, 3: 6}[r['level']]), r['name'])
                      for r in load(args.official / 'mca-current-flat.json')
                      if r.get('code') and int(r['code']) < 700000}
    selected = [r for r in metadata['relations'] if any((r['tags'].get('admin_level'), n) in official_names
                for n in names(r['tags']))]
    pbf = extract / 'administrative.osm.pbf'
    if args.stage == 'audit':
        way_ids = {w.id for w in osmium.FileProcessor(str(pbf), osmium.osm.WAY)}
        incomplete = []
        for relation in selected:
            missing = [m['ref'] for m in relation['members'] if m['type'] == 'w'
                       and m['role'] in {'outer', 'inner', ''} and m['ref'] not in way_ids]
            if missing:
                incomplete.append({'id': relation['id'], 'name': relation['tags'].get('name'),
                                   'adminLevel': relation['tags'].get('admin_level'), 'missingWays': missing})
        audit = {'snapshot': metadata['snapshot'], 'selectedRelations': len(selected),
                 'incompleteRelations': incomplete, 'missingWays': sorted({w for r in incomplete for w in r['missingWays']}),
                 'selection': 'exact official name and source administrative level; no fuzzy or nearest matching'}
        save(output / 'completion-plan.json', audit)
        print(json.dumps({'selected': len(selected), 'incomplete': len(incomplete),
                          'levels': dict(Counter(r['adminLevel'] for r in incomplete)),
                          'missingWays': len(audit['missingWays'])}, ensure_ascii=False), flush=True)
    elif args.stage == 'fetch':
        plan = load(output / 'completion-plan.json')
        results = []
        for offset in range(0, len(plan['missingWays']), args.batch_size):
            ids = plan['missingWays'][offset:offset + args.batch_size]
            query = f'[out:json][timeout:55][date:"{plan["snapshot"]}"];way(id:' + ','.join(map(str, ids)) + ');out body geom;'
            result = acquire(output / 'responses', f'ways-{offset:05d}.json',
                             args.endpoint + '?' + urlencode({'data': query}), query=query)
            results.append(result)
            save(output / 'completion-fetch.json', {'results': results})
            if 'error' in result:
                raise SystemExit(1)
    elif args.stage == 'fetch-api':
        plan = load(output / 'completion-plan.json')
        # A small explicit missing-object set, not bulk extraction from the
        # editing API. A current object is usable at the frozen timestamp only
        # when it and every referenced node were last modified before it.
        def fetch(identifier):
            folder = output / 'api-responses'
            name = f'way-{identifier}-full.json'
            result = acquire(folder, name,
                             f'https://api.openstreetmap.org/api/0.6/way/{identifier}/full.json', timeout=35,
                             limit=3_000_000)
            if 'error' in result:
                return result
            elements = load(folder / name)['elements']
            newer = [{'type': e['type'], 'id': e['id'], 'timestamp': e.get('timestamp')}
                     for e in elements if not e.get('timestamp') or e['timestamp'] > plan['snapshot']]
            source_ways = [e for e in elements if e['type'] == 'way' and e['id'] == identifier]
            source_nodes = {e['id'] for e in elements if e['type'] == 'node'}
            if len(source_ways) != 1 or not set(source_ways[0]['nodes']) <= source_nodes:
                result['error'] = 'Incomplete official full-way response'
            elif newer:
                result['error'] = 'Objects newer than the frozen snapshot; historical versions required'
                result['newerObjects'] = newer
            else:
                result['verifiedSnapshot'] = plan['snapshot']
                result['maximumObjectTimestamp'] = max(e['timestamp'] for e in elements)
                result['verification'] = 'Every object last-modified timestamp <= frozen snapshot'
            save(folder / (name + '.snapshot-proof.json'), result)
            return result
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(fetch, plan['missingWays']))
        save(output / 'completion-api-fetch.json', {'snapshot': plan['snapshot'], 'results': results})
        if any('error' in r for r in results):
            raise SystemExit(1)
    elif args.stage == 'supplement-relations':
        from shapely.geometry import shape
        if not args.additional_relation:
            parser.error('Choose explicitly reviewed additional relation responses')
        completed = output / 'administrative-complete.osm.pbf'
        if hashlib.sha256(completed.read_bytes()).hexdigest() != load(output / 'completion-receipt.json')['sha256']:
            raise ValueError('Completed PBF hash mismatch')
        subset = output / 'additional-relations.osm.pbf'
        if subset.exists():
            raise ValueError('Additional relation subset already exists')
        relations, inputs = {}, []
        with osmium.BackReferenceWriter(str(subset), ref_src=str(completed), relation_depth=0) as writer:
            for path in args.additional_relation:
                receipt = load(path.with_suffix(path.suffix + '.receipt.json'))
                if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']:
                    raise ValueError('Additional relation response hash mismatch')
                elements = load(path)['elements']
                if len(elements) != 1 or elements[0]['type'] != 'relation' or elements[0]['timestamp'] > metadata['snapshot']:
                    raise ValueError('Additional relation is not a verified frozen relation')
                r = elements[0]
                normalized = {'id': r['id'], 'version': r['version'], 'timestamp': r['timestamp'], 'tags': r['tags'],
                              'members': [{**m, 'type': {'way': 'w', 'node': 'n', 'relation': 'r'}[m['type']]} for m in r['members']]}
                relations[r['id']] = normalized
                members = [(m['type'], m['ref'], m['role']) for m in normalized['members']
                           if m['type'] in {'w', 'r'} and m['role'] in {'outer', 'inner', ''}]
                writer.add(osmium.osm.mutable.Relation(id=r['id'], version=r['version'], tags=r['tags'], members=members))
                inputs.append(receipt | {'verifiedSnapshot': metadata['snapshot']})
        factory, features = osmium.geom.GeoJSONFactory(), []
        for area in osmium.FileProcessor(str(subset)).with_locations('flex_mem').with_areas():
            if not area.is_area() or area.from_way() or area.orig_id() not in relations:
                continue
            r = relations[area.orig_id()]
            geometry = json.loads(factory.create_multipolygon(area))
            if shape(geometry).is_empty or not shape(geometry).is_valid:
                raise ValueError('Additional relation geometry is invalid')
            features.append({'type': 'Feature', 'id': f'osm:relation:{r["id"]}', 'geometry': geometry,
                             'properties': {'osmId': r['id'], 'osmVersion': r['version'], 'snapshot': metadata['snapshot'], 'tags': r['tags']}})
        if len(features) != len(relations):
            raise ValueError('Additional relations have missing geometry members')
        save(output / 'additional-boundaries.geojson', {'type': 'FeatureCollection', 'features': features})
        save(output / 'additional-relations.json', {'snapshot': metadata['snapshot'], 'relations': list(relations.values()), 'inputs': inputs})
        print(json.dumps({'additionalRelations': len(relations), 'validPolygons': len(features), 'osmIds': sorted(relations)}, ensure_ascii=False), flush=True)
    elif args.stage == 'polygonize':
        from shapely.geometry import shape
        receipt = load(output / 'completion-receipt.json')
        completed = output / 'administrative-complete.osm.pbf'
        if hashlib.sha256(completed.read_bytes()).hexdigest() != receipt['sha256']:
            raise ValueError('Completed PBF hash mismatch')
        relations = {r['id']: r for r in metadata['relations']}
        factory = osmium.geom.GeoJSONFactory()
        features, invalid, seen = [], [], set()
        for area in osmium.FileProcessor(str(completed)).with_locations('flex_mem').with_areas():
            if not area.is_area() or area.from_way() or area.orig_id() not in relations:
                continue
            identifier = area.orig_id()
            seen.add(identifier)
            geometry = json.loads(factory.create_multipolygon(area))
            polygon = shape(geometry)
            if polygon.is_empty or not polygon.is_valid or polygon.geom_type not in {'Polygon', 'MultiPolygon'}:
                invalid.append({'osmId': identifier, 'reason': 'empty/invalid/nonpolygon'})
                continue
            relation = relations[identifier]
            features.append({'type': 'Feature', 'id': f'osm:relation:{identifier}',
                             'properties': {'osmId': identifier, 'osmVersion': relation['version'],
                                            'snapshot': metadata['snapshot'], 'tags': relation['tags']},
                             'geometry': geometry})
        target = output / 'boundaries.geojson'
        if target.exists():
            raise ValueError('Completed geometry already exists; never overwrite an audited candidate')
        save(target, {'type': 'FeatureCollection', 'features': features})
        audit = {'snapshot': metadata['snapshot'], 'validPolygons': len(features), 'invalid': invalid,
                 'unassembled': [{'osmId': r['id'], 'name': r['tags'].get('name'),
                                  'adminLevel': r['tags'].get('admin_level')}
                                 for r in metadata['relations'] if r['id'] not in seen],
                 'completion': receipt, 'bytes': target.stat().st_size,
                 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        save(output / 'polygon-audit.json', audit)
        print(json.dumps({k: v for k, v in audit.items() if k not in {'completion', 'unassembled'}}, ensure_ascii=False), flush=True)
    else:
        plan = load(output / 'completion-plan.json')
        nodes, ways = {}, {}
        responses = []
        paths = [p for p in (output / 'responses').glob('ways-*.json') if re.fullmatch(r'ways-\d+\.json', p.name)]
        paths += list((output / 'api-responses').glob('way-*-full.json'))
        for path in sorted(paths):
            receipt = load(path.with_suffix(path.suffix + '.receipt.json'))
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']:
                raise ValueError('Supplement response hash mismatch')
            if path.parent.name == 'api-responses':
                proof = load(path.with_suffix(path.suffix + '.snapshot-proof.json'))
                if proof.get('error') or proof.get('verifiedSnapshot') != plan['snapshot'] or proof.get('sha256') != receipt['sha256']:
                    raise ValueError('Official API object snapshot proof failed')
                if any(not e.get('timestamp') or e['timestamp'] > plan['snapshot'] for e in load(path)['elements']):
                    raise ValueError('Official API object is newer than snapshot')
            elif plan['snapshot'] not in receipt['query']:
                raise ValueError('Supplement response snapshot mismatch')
            responses.append(receipt)
            for element in load(path)['elements']:
                if element['type'] == 'node':
                    existing = nodes.get(element['id'])
                    if existing and (existing['lon'], existing['lat']) != (element['lon'], element['lat']):
                        raise ValueError('Conflicting supplement node coordinates')
                    nodes[element['id']] = element
                elif element['type'] == 'way':
                    ways[element['id']] = element
                    if path.parent.name == 'api-responses':
                        continue
                    if len(element.get('nodes', [])) != len(element.get('geometry', [])):
                        raise ValueError('Missing full way geometry')
                    for identifier, coordinate in zip(element['nodes'], element['geometry']):
                        existing = nodes.get(identifier)
                        if existing and (existing['lon'], existing['lat']) != (coordinate['lon'], coordinate['lat']):
                            raise ValueError('Conflicting supplement node coordinates')
                        nodes[identifier] = {'id': identifier, **coordinate}
        missing = set(plan['missingWays']) - ways.keys()
        if missing:
            raise ValueError('Incomplete downloaded way set: ' + str(sorted(missing)))
        path = output / 'administrative-complete.osm.pbf'
        if path.exists():
            raise ValueError('Completed PBF already exists; never overwrite an audited candidate')
        counters = Counter()
        with osmium.SimpleWriter(str(path)) as writer:
            # Original source order is preserved within each type. Supplemental
            # objects are merged by ID and may only fill absent objects.
            for kind, values in [('node', nodes), ('way', ways)]:
                ids = iter(sorted(values))
                pending = next(ids, None)
                bits = osmium.osm.NODE if kind == 'node' else osmium.osm.WAY
                def add(identifier):
                    e = values[identifier]
                    if kind == 'node':
                        writer.add_node(osmium.osm.mutable.Node(id=identifier, location=(e['lon'], e['lat']), tags=e.get('tags', {})))
                    else:
                        writer.add_way(osmium.osm.mutable.Way(id=identifier, nodes=e['nodes'], tags=e.get('tags', {})))
                    counters['added-' + kind] += 1
                for original in osmium.FileProcessor(str(pbf), bits):
                    while pending is not None and pending < original.id:
                        add(pending)
                        pending = next(ids, None)
                    if pending == original.id:
                        if kind == 'node' and (original.location.lon != values[pending]['lon'] or original.location.lat != values[pending]['lat']):
                            raise ValueError('Source and supplement disagree at a shared node')
                        pending = next(ids, None)
                    writer.add(original)
                    counters['original-' + kind] += 1
                while pending is not None:
                    add(pending)
                    pending = next(ids, None)
            for relation in osmium.FileProcessor(str(pbf), osmium.osm.RELATION):
                writer.add_relation(relation)
                counters['original-relation'] += 1
        receipt = {'snapshot': plan['snapshot'], 'source': str(pbf), 'counters': dict(counters),
                   'responses': responses, 'bytes': path.stat().st_size,
                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'operation': 'fill absent ways/nodes only'}
        save(output / 'completion-receipt.json', receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k != 'responses'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
