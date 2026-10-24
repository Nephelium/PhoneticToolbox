"""Boundary rebuild checks. Run with the task-local Shapely tools on sys.path."""
import importlib.util
import copy
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('open_maps', Path(__file__).resolve().parents[1] / 'scripts/prepare_geolinguistics_open_maps.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def polygon(x0, y0, x1, y1):
    return {'type': 'Polygon', 'coordinates': [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def source(rid, name, level, geometry, **tags):
    return {'type': 'Feature', 'geometry': geometry, 'properties': {'osmId': rid, 'osmVersion': 1,
        'tags': {'name': name, 'admin_level': str(level), **tags}}}


class OpenMapIdentity(unittest.TestCase):
    def setUp(self):
        self.official = [
            {'code': '440000', 'name': '广东省', 'level': 1, 'parentCode': '00'},
            {'code': '440100', 'name': '甲市', 'level': 2, 'parentCode': '44'},
            {'code': '440200', 'name': '乙市', 'level': 2, 'parentCode': '44'},
            {'code': '440101', 'name': '同名区', 'level': 3, 'parentCode': '4401'},
            {'code': '440201', 'name': '同名区', 'level': 3, 'parentCode': '4402'},
        ]
        self.osm = [source(1, '广东省', 4, polygon(0, 0, 10, 10), **{'ISO3166-2': 'CN-GD'}),
                    source(2, '甲市', 5, polygon(0, 0, 5, 10)),
                    source(3, '乙市', 5, polygon(5, 0, 10, 10)),
                    source(4, '同名区', 6, polygon(0, 0, 5, 10)),
                    source(5, '同名区', 6, polygon(5, 0, 10, 10))]

    def test_normalizes_only_explicit_administrative_code_forms(self):
        self.assertEqual(module.explicit_codes({'ref:admin:CN': '44;440000', 'ref:admin:CN:mca': '440100000'}), {'440000', '440100'})
        self.assertEqual(module.explicit_codes({'ref:admin:CN': '440101123;guess-440100'}), set())

    def test_same_name_counties_require_actual_parent_polygon(self):
        _, audit = module.mainland(self.osm, self.official)
        self.assertEqual(audit['accepted'], 5)
        mapping = {r['osmId']: r['code'] for r in audit['evidence']}
        self.assertEqual(mapping[4], '440101')
        self.assertEqual(mapping[5], '440201')

    def test_nearby_outside_polygon_is_not_associated(self):
        self.osm[4]['geometry'] = polygon(10.001, 0, 11, 1)
        _, audit = module.mainland(self.osm, self.official)
        self.assertNotIn(5, {r['osmId'] for r in audit['evidence']})
        self.assertIn('440201', {r['code'] for r in audit['missing']})

    def test_conflicting_explicit_code_is_not_overridden_by_name(self):
        self.osm[3]['properties']['tags']['ref:admin:CN'] = '440201'
        _, audit = module.mainland(self.osm, self.official)
        self.assertNotIn(4, {r['osmId'] for r in audit['evidence']})

    def test_duplicate_relations_cannot_silently_select_first(self):
        self.osm.append(source(6, '甲市', 5, polygon(0, 0, 5, 10)))
        _, audit = module.mainland(self.osm, self.official)
        self.assertIn({'code': '440100', 'osmIds': [2, 6], 'reason': 'duplicate-relations'}, audit['pending'])
        self.assertNotIn('440100', {r['code'] for r in audit['evidence']})

    def test_reviewed_legacy_code_still_requires_exact_identity_and_geometry(self):
        self.osm[3]['properties']['tags']['ref:admin:CN'] = '440199'
        review = {'entries': [{'osmId': 4, 'osmVersion': 1, 'sourceName': '同名区',
                               'officialName': '同名区', 'officialCode': '440101',
                               'nativeCode': {'key': 'ref:admin:CN', 'value': '440199'},
                               'legacyCodes': ['440199']}]}
        _, audit = module.mainland(self.osm, self.official, correspondence=review)
        self.assertIn('440101', {r['code'] for r in audit['evidence']})
        self.osm[3]['geometry'] = polygon(11, 0, 12, 1)
        _, audit = module.mainland(self.osm, self.official, correspondence=review)
        self.assertNotIn('440101', {r['code'] for r in audit['evidence']})
        self.osm[3]['properties']['osmVersion'] = 2
        with self.assertRaisesRegex(ValueError, 'Reviewed OSM identity changed'):
            module.mainland(self.osm, self.official, correspondence=review)

    def test_bilingual_correspondence_is_explicit_and_version_pinned(self):
        self.osm[3]['properties']['tags'].update(name='同名区 བོད།', division_code='440101')
        _, audit = module.mainland(self.osm, self.official)
        self.assertNotIn('440101', {r['code'] for r in audit['evidence']})
        review = {'entries': [{'osmId': 4, 'osmVersion': 1, 'sourceName': '同名区 བོད།',
                               'officialName': '同名区', 'officialCode': '440101',
                               'nativeCode': {'key': 'division_code', 'value': '440101'}}]}
        _, audit = module.mainland(self.osm, self.official, correspondence=review)
        self.assertIn('440101', {r['code'] for r in audit['evidence']})
        self.assertEqual(audit['evidence'][3]['reviewedCorrespondence']['officialName'], '同名区')

    def test_source_subarea_can_identify_an_exclave_without_certifying_its_geometry(self):
        self.osm[4]['geometry'] = polygon(10.001, 0, 11, 1)
        relations = {'relations': [{'id': 1, 'members': [{'type': 'r', 'ref': 3, 'role': 'subarea'}]},
                                    {'id': 3, 'members': [{'type': 'r', 'ref': 5, 'role': 'subarea'}]}]}
        _, audit = module.mainland(self.osm, self.official, relations=relations)
        row = next(r for r in audit['evidence'] if r['osmId'] == 5)
        self.assertEqual(row['code'], '440201')
        self.assertEqual(row['sourceSubareaPath'], [1, 3, 5])
        self.assertEqual(row['provinceCoveredFraction'], 0)
        self.assertIn('review-required', row['geometryStatus'])

    def test_reviewed_union_preserves_raw_source_and_rejects_changed_components(self):
        original=copy.deepcopy(self.osm)
        review={'geometryDerivations':[{'officialCode':'440101','officialName':'同名区',
            'target':{'osmId':4,'osmVersion':1,'sourceName':'同名区'},
            'components':[{'osmId':4,'osmVersion':1,'sourceName':'同名区'},
                          {'osmId':5,'osmVersion':1,'sourceName':'同名区'}],
            'evidence':'reviewed source grouping','urls':['https://example.test/review']} ]}
        rebuilt,audit=module.apply_geometry_derivations(self.osm,review)
        from shapely.geometry import shape
        self.assertEqual(shape(rebuilt[3]['geometry']).area,100)
        self.assertEqual(self.osm,original)
        self.assertEqual(audit[0]['removedAreaDegrees2'],0)
        self.osm[4]['properties']['osmVersion']=2
        with self.assertRaisesRegex(ValueError,'component version/name changed'):
            module.apply_geometry_derivations(self.osm,review)

    def test_city_without_county_children_has_explicit_display_area(self):
        features, audit = module.mainland(self.osm[:3], self.official[:3])
        area = next(f for f in features if f['id'] == 'district:440100')
        self.assertTrue(area['properties']['displayAggregate'])
        self.assertEqual(area['properties']['officialLevel'], 2)
        self.assertEqual(audit['accepted'], 3)

    def test_point_boundary_ambiguity_and_raw_values_remain(self):
        features = [module.feature('district', '1', '甲', 'city:1', 'province:1', polygon(0, 0, 1, 1), 'test'),
                    module.feature('district', '2', '乙', 'city:1', 'province:1', polygon(1, 0, 2, 1), 'test')]
        sites = [{'id': 'S1', 'name': '内部', 'lon': .5, 'lat': .5, 'admin': 'district:1'},
                 {'id': 'S2', 'name': '边界', 'lon': 1, 'lat': .5, 'admin': 'district:1'},
                 {'id': 'S3', 'name': '外部', 'lon': 2.01, 'lat': .5, 'admin': 'district:2'}]
        rows = [{'id': 'r1', 'admin': 'district:1', 'values': ['原值'], 'original': ['原值']},
                {'id': 'r2', 'admin': 'district:2', 'values': ['旧名'], 'original': ['旧名']}]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for name, data in [('sites.json', sites), ('classification.json', {'sheets': [{'id': 'x', 'rows': rows}]}),
                               ('map-index.json', [{'id': 'district:1', 'name': '甲'}, {'id': 'district:2', 'name': '改名前'}])]:
                (folder / name).write_text(json.dumps(data), encoding='utf-8')
            new_sites, classes, audit = module.audit_associations(features, folder)
        self.assertEqual([s['admin'] for s in new_sites], ['district:1', None, None])
        self.assertEqual(new_sites[1]['adminCandidates'], ['district:1', 'district:2'])
        self.assertEqual([s['lon'] for s in new_sites], [.5, 1, 2.01])
        self.assertEqual(classes['sheets'][0]['rows'][0]['original'], ['原值'])
        self.assertIsNone(classes['sheets'][0]['rows'][1]['admin'])
        self.assertEqual(audit['coordinateChanges'], 0)


if __name__ == '__main__':
    unittest.main()
