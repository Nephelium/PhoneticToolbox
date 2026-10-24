"""Source-native geometry normalization against small independent examples."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verification_artifacts import verification_run
from prepare_geolinguistics_datav import atomic, digest, encoded

spec = importlib.util.spec_from_file_location('datav_normalize', ROOT / 'scripts/normalize_geolinguistics_datav.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def square(x, y, size=2):
    return [[[x, y], [x+size, y], [x+size, y+size], [x, y+size], [x, y]]]


class DatavNormalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact_run = verification_run('geolinguistics-datav-normalize', 'unit', 'Synthetic overlap, ring, collapse, tree, source selection and reproducibility; actual raw candidate separate')
        cls.out, cls.scratch = cls.artifact_run.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.artifact_run.__exit__(None, None, None)

    def test_overlapping_polygon_members_keep_the_overlap_filled(self):
        from shapely.geometry import shape, Point
        raw = {'type': 'MultiPolygon', 'coordinates': [square(0, 0), square(1, 1)]}
        result, audit, collapsed = module.normalize_geometry(raw)
        geometry = shape(result)
        self.assertEqual(geometry.area, 7)
        self.assertTrue(geometry.covers(Point(1.5, 1.5)))
        self.assertTrue(geometry.is_valid)
        self.assertEqual(audit['before']['areaNativeUnitsSquared'], 8)
        self.assertEqual(audit['sumRepairedAreaMinusUnionAreaNativeUnitsSquared'], 1)
        self.assertEqual({tuple(r['coordinate']) for r in audit['addedVertices']}, {(1, 2), (2, 1)})
        self.assertTrue(all(r['intersectionSupported'] for r in audit['addedVertices']))
        self.assertEqual(collapsed, [])

    def test_self_crossing_polygon_records_the_new_intersection_and_all_parts(self):
        from shapely.geometry import shape
        raw = {'type': 'Polygon', 'coordinates': [[[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]]]}
        result, audit, _ = module.normalize_geometry(raw)
        self.assertEqual(shape(result).area, 2)
        self.assertEqual(audit['after']['polygonParts'], 2)
        self.assertEqual(audit['addedVertices'][0]['coordinate'], [1, 1])
        self.assertIn('Self-intersection', audit['before']['validity'])
        self.assertIn('resultGeometry', audit['polygonMembers'][0])

    def test_valid_shell_hole_coordinates_remain_exact(self):
        raw = {'type': 'Polygon', 'coordinates': square(0, 0, 4) + square(1, 1, 2)}
        original = copy.deepcopy(raw)
        result, audit, collapsed = module.normalize_geometry(raw)
        self.assertEqual(result, original)
        self.assertEqual(audit['after']['areaNativeUnitsSquared'], 12)
        self.assertEqual(audit['after']['holes'], 1)
        self.assertFalse(audit['repaired'])
        self.assertEqual(collapsed, [])
        self.assertEqual(raw, original)

    def test_collapsed_line_and_point_are_retained_separately_from_administrative_area(self):
        from shapely.geometry import shape
        raw = {'type': 'MultiPolygon', 'coordinates': [square(0, 0),
               [[[5, 0], [6, 0], [7, 0], [5, 0]]], [[[9, 0], [9, 0], [9, 0], [9, 0]]]]}
        result, audit, collapsed = module.normalize_geometry(raw)
        self.assertEqual(shape(result).area, 4)
        self.assertEqual({part['geometry']['type'] for part in collapsed}, {'LineString', 'Point'})
        self.assertEqual(len(audit['collapsedComponents']), 2)
        self.assertEqual({part['polygonIndex'] for part in collapsed}, {1, 2})
        self.assertEqual(next(part for part in collapsed if part['geometry']['type'] == 'Point')['geometry']['coordinates'], (9.0, 0.0))

    def test_tree_keeps_direct_parent_source_city_leaf_and_taiwan_without_synthetic_nodes(self):
        tree = {n['adcode']: n for n in [
            {'adcode': 100000, 'level': 'country', 'parent': None},
            {'adcode': 110000, 'level': 'province', 'parent': 100000},
            {'adcode': 110101, 'level': 'district', 'parent': 110000},
            {'adcode': 420000, 'level': 'province', 'parent': 100000},
            {'adcode': 429004, 'level': 'city', 'parent': 420000},
            {'adcode': 710000, 'level': 'province', 'parent': 100000}]}
        snapshot = copy.deepcopy(tree)
        specials, children = module.special_tree(tree)
        self.assertEqual(len(specials['districtsDirectlyUnderProvince']), 1)
        self.assertEqual(specials['provinceDirectLeafEntitiesSourceLabelsCityNon00Code'][0]['adcode'], 429004)
        self.assertEqual(children[110000], [110101])
        self.assertEqual(specials['taiwan']['children'], [])
        self.assertEqual(tree, snapshot)

    def make_input(self, directory):
        directory.mkdir()
        tree = [{'adcode': 100000, 'name': '中国', 'level': 'country', 'parent': None},
                {'adcode': 710000, 'name': '台湾省', 'level': 'province', 'parent': 100000}]
        feature = lambda geometry: {'type': 'Feature', 'properties': {'adcode': 710000, 'name': '台湾省', 'level': 'province', 'parent': {'adcode': 100000}}, 'geometry': geometry}
        full = feature({'type': 'MultiPolygon', 'coordinates': [square(0, 0), square(5, 5)]})
        outline = feature({'type': 'MultiPolygon', 'coordinates': [square(0, 0)]})
        symbol = {'type': 'Feature', 'properties': {'adcode': '100000_JD'}, 'geometry': {'type': 'MultiPolygon', 'coordinates': [square(10, 10)]}}
        values = {'all.json': tree, '100000_full.json': {'type': 'FeatureCollection', 'features': [full, symbol]},
                  '710000.json': {'type': 'FeatureCollection', 'features': [outline]}}
        records = []
        for name, value in values.items():
            raw = encoded(value);(directory / name).write_bytes(raw)
            row = {'name': name, 'status': 200, 'sha256': digest(raw)};records.append(row)
            atomic(module.receipt_path(directory, name) if hasattr(module, 'receipt_path') else directory / (name.removesuffix('.json') + '.receipt.json'), encoded(row))
        records.sort(key=lambda r: r['name'])
        source_set = digest(encoded({r['name']: r['sha256'] for r in records}))
        atomic(directory / 'source-inventory.json', encoded({'files': records, 'rawSourceSetSha256': source_set}))
        return source_set, values

    def test_national_province_source_preserves_all_island_parts_and_build_is_reproducible(self):
        source = self.scratch / 'raw'
        source_set, original = self.make_input(source)
        index = module.tree_index(original['all.json'])
        chosen, auxiliary = module.select_sources(source, index)
        self.assertEqual(chosen[710000][0]['file'], '100000_full.json')
        self.assertEqual(len(chosen[710000][1]['geometry']['coordinates']), 2)
        self.assertEqual(len(auxiliary), 1)
        first = module.build_candidate(source, self.scratch / 'first', source_set)
        second = module.build_candidate(source, self.scratch / 'second', source_set)
        self.assertEqual(first, second)
        self.assertEqual(first['counts'], {'province': 1, 'city': 0, 'district': 0})
        self.assertEqual(json.loads((source / '100000_full.json').read_bytes()), original['100000_full.json'])
        self.assertEqual(first['repairedAdministrativeFeatures'], 0)

    def test_modified_raw_input_is_rejected_before_output_is_created(self):
        source = self.scratch / 'corrupt'
        source_set, _ = self.make_input(source)
        (source / '710000.json').write_bytes(b'changed')
        target = self.scratch / 'not-created'
        with self.assertRaisesRegex(ValueError, 'Raw input SHA'):
            module.build_candidate(source, target, source_set)
        self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
