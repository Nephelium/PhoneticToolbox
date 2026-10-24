"""Bounded GeoAtlas acquisition tests; HTTP stand-ins are not source evidence."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verification_artifacts import verification_run

spec = importlib.util.spec_from_file_location('datav_candidate', ROOT / 'scripts/prepare_geolinguistics_datav.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Response(io.BytesIO):
    def __init__(self, raw, status=200, name='440000_full.json'):
        super().__init__(raw)
        self.code, self.name, self.headers = status, name, {'Content-Type': 'application/json'}

    def geturl(self):
        return module.BASE + self.name


class DatavAcquisition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact_run = verification_run('geolinguistics-datav', 'unit', 'Synthetic bounded HTTP/cache/tree/geometry cases; actual source acquisition recorded separately')
        cls.out, cls.scratch = cls.artifact_run.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.artifact_run.__exit__(None, None, None)

    def setUp(self):
        self.output = self.scratch / self._testMethodName
        self.output.mkdir()
        self.client = module.Acquisition(self.output, delay=0)

    def raw(self, features=None):
        return module.encoded({'type': 'FeatureCollection', 'features': features or []})

    def index(self):
        return module.tree_index([
            {'adcode': 100000, 'level': 'country', 'parent': None},
            {'adcode': 440000, 'level': 'province', 'parent': 100000},
            {'adcode': 440100, 'level': 'city', 'parent': 440000},
            {'adcode': 440101, 'level': 'district', 'parent': 440100},
            {'adcode': 110000, 'level': 'province', 'parent': 100000},
            {'adcode': 710000, 'level': 'province', 'parent': 100000}])

    def test_only_official_tree_codes_and_selector_suffixes_are_planned(self):
        self.assertEqual(module.aggregate_plan(self.index()), ['100000_full_city.json', '110000_full.json', '440000_full.json', '440000_full_district.json'])
        with self.assertRaises(ValueError):
            self.client.acquire('../secret.json')
        self.assertIsNone(module.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://other.invalid/'))
        target = self.output / 'guard.json'
        with patch.object(Path, 'is_symlink', autospec=True, side_effect=lambda path: path == target.with_suffix('.json.part')):
            with self.assertRaisesRegex(ValueError, 'linked output'):
                module.atomic(target, b'never written')
        self.assertFalse(target.exists())

    def test_duplicate_missing_and_cyclic_tree_parents_fail(self):
        for tree in ([{'adcode': 100000, 'level': 'country', 'parent': None}] * 2,
                     [{'adcode': 100000, 'level': 'country', 'parent': 440000}],
                     [{'adcode': 100000, 'level': 'country', 'parent': 100000}]):
            with self.assertRaises(ValueError):
                module.tree_index(tree)

    def test_seed_and_resume_preserve_exact_bytes_and_cached_404(self):
        source = self.output / 'seed'
        source.mkdir()
        raw = b'{"type":"FeatureCollection", "features":[]}\n'
        (source / '440000_full.json').write_bytes(raw)
        records = [{'name': '440000_full', 'url': module.BASE + '440000_full.json', 'downloaded': True, 'bytes': len(raw), 'sha256': module.digest(raw)},
                   {'name': '710000_full', 'url': module.BASE + '710000_full.json', 'status': 404, 'responseSha256': 'kept'}]
        self.client.seed(source, records)
        with patch.object(module, 'build_opener', side_effect=AssertionError('No repeat GET')):
            self.assertEqual(self.client.acquire('440000_full.json')['sha256'], module.digest(raw))
            self.assertEqual(self.client.acquire('710000_full.json')['status'], 404)
        self.assertEqual((self.output / '440000_full.json').read_bytes(), raw)
        (self.output / '440000_full.json').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Cached bytes'):
            self.client.acquire('440000_full.json')

    def test_transient_retry_is_finite_and_success_is_receipted(self):
        responses = [Response(b'busy', 503), Response(self.raw())]
        class Opener:
            def open(inner, request, timeout):
                self.assertEqual(request.get_method(), 'GET')
                self.assertEqual(timeout, 25)
                return responses.pop(0)
        with patch.object(module, 'build_opener', return_value=Opener()), patch.object(module.time, 'sleep'):
            record = self.client.acquire('440000_full.json')
        self.assertEqual([r['status'] for r in record['attempts']], [503, 200])
        self.assertEqual(record['sha256'], module.digest(self.raw()))
        self.assertEqual(self.client.network_attempts, 2)

    def test_404_body_is_kept_without_retries_and_invalid_json_never_becomes_input(self):
        class Opener:
            def open(inner, request, timeout):
                return Response(b'<NoSuchKey/>', 404)
        with patch.object(module, 'build_opener', return_value=Opener()):
            self.assertEqual(self.client.acquire('440000_full.json')['status'], 404)
        self.assertEqual(self.client.network_attempts, 1)
        self.assertEqual((self.output / '440000_full.json.http404.body').read_bytes(), b'<NoSuchKey/>')
        class Invalid:
            def open(inner, request, timeout):
                return Response(b'<html>not geometry</html>')
        with patch.object(module, 'build_opener', return_value=Invalid()):
            record = self.client.acquire('440100_full.json')
        self.assertIsNone(record['status'])
        self.assertFalse((self.output / '440100_full.json').exists())
        self.assertEqual((self.output / '440100_full.json.invalid-response.body').read_bytes(), b'<html>not geometry</html>')

    def test_batch_never_exceeds_two_concurrent_requests(self):
        active = maximum = 0
        lock = threading.Lock()
        raw = self.raw()
        class Opener:
            def open(inner, request, timeout):
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(active, maximum)
                time.sleep(.02)
                with lock:
                    active -= 1
                return Response(raw, name=request.full_url.rsplit('/', 1)[1])
        with patch.object(module, 'build_opener', return_value=Opener()):
            records = self.client.batch([str(code) + '.json' for code in range(440101, 440107)])
        self.assertEqual(maximum, 2)
        self.assertEqual(self.client.network_attempts, 6)
        self.assertTrue(all(r['status'] == 200 for r in records))

    def test_raw_symbols_and_invalid_polygons_are_preserved_and_separately_audited(self):
        polygon = {'type': 'Polygon', 'coordinates': [[[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]]]}
        features = [{'type': 'Feature', 'properties': {'adcode': 440000, 'level': 'province'}, 'geometry': polygon},
                    {'type': 'Feature', 'properties': {'adcode': '100000_JD'}, 'geometry': polygon}]
        raw = self.raw(features)
        (self.output / '100000_full.json').write_bytes(raw)
        module.atomic(module.receipt_path(self.output, '100000_full.json'), module.encoded({'name': '100000_full.json', 'url': module.BASE + '100000_full.json', 'status': 200, 'sha256': module.digest(raw)}))
        coverage = module.coverage(self.output, self.index())
        self.assertEqual(coverage['covered'], {'province': 1})
        self.assertEqual(coverage['extraAndSymbols'][0]['adcode'], '100000_JD')
        report = module.audit(self.output, self.index())
        self.assertEqual(len(report['files']['100000_full.json']['invalid']), 2)
        self.assertEqual(report['layerUniqueCodes']['province'], 1)
        self.assertEqual((self.output / '100000_full.json').read_bytes(), raw)


if __name__ == '__main__':
    unittest.main()
