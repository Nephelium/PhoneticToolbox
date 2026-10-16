"""Cache integrity regressions: bytes, source identity, recipe and corruption."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/manual'))
sys.path.insert(0, str(ROOT / 'scripts'))
from pack_media import package_asset
from package_local import select_cache


class MediaCacheTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / 'output/validation/packaging-cache-tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        from PIL import Image
        self.source = self.root / 'source.png'
        Image.new('RGB', (320, 240), '#88aacc').save(self.source)
        self.asset = {'id': 'image', 'kind': 'image', 'mime': 'image/png',
                      'sha256': hashlib.sha256(self.source.read_bytes()).hexdigest()}
        self.cache = self.root / 'cache'

    def test_reuse_preserves_bytes_and_reader_metadata_without_encoding(self):
        first, initial = package_asset(dict(self.asset), self.source, self.root / 'first', self.cache)
        with patch('PIL.Image.Image.save', side_effect=AssertionError('must reuse checked bytes')):
            second, repeat = package_asset(dict(self.asset), self.source, self.root / 'second', self.cache)
        self.assertEqual(first, second)
        self.assertEqual((self.root / 'first' / first['path']).read_bytes(), (self.root / 'second' / second['path']).read_bytes())
        self.assertFalse(initial['cacheHit'])
        self.assertTrue(repeat['cacheHit'])

    def test_corrupt_blob_and_changed_source_are_rejected(self):
        package_asset(dict(self.asset), self.source, self.root / 'first', self.cache)
        next(self.cache.glob('*.bin')).write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'cache identity'):
            package_asset(dict(self.asset), self.source, self.root / 'second', self.cache)
        self.source.write_bytes(b'different source')
        with self.assertRaisesRegex(ValueError, 'source media hash'):
            package_asset(dict(self.asset), self.source, self.root / 'third', self.cache)

    def test_changed_codec_requires_new_derivative(self):
        package_asset(dict(self.asset), self.source, self.root / 'first', self.cache)
        with patch('pack_media.encoder_identity', return_value={'webp': 'changed'}):
            _, result = package_asset(dict(self.asset), self.source, self.root / 'second', self.cache)
        self.assertFalse(result['cacheHit'])
        self.assertEqual(len(list(self.cache.glob('*.json'))), 2)

    def test_incomplete_cache_is_never_selected(self):
        for name, state, time in [('old', 'completed', '2026-10-12'), ('new', 'failed', '2026-10-13')]:
            work = self.root / 'output' / ('build-' + name)
            (work / 'snapshot').mkdir(parents=True)
            (work / 'host-archive').mkdir()
            (work / 'host-archive/host-archive.json').write_text('{}')
            (work / 'artifact-lifecycle.json').write_text(json.dumps({'state': state, 'finished_at': time}))
            (work / 'snapshot/desktop-bundle.json').write_text(json.dumps({'runtimes': {'PTB_EGG_PYTHON': {}, 'PTB_M05_PYTHON': {}}}))
            dist = self.root / 'dist' / name
            dist.mkdir(parents=True)
            (dist / 'build-info.json').write_text(json.dumps({'portable': True, 'archives': {'host': {}}}))
        self.assertEqual(select_cache(self.root).name, 'build-old')


if __name__ == '__main__':
    unittest.main()
