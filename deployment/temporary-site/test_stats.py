"""Focused regression checks for counting, resume requests and log rotation."""
from pathlib import Path
import gzip
import json
import tempfile
import unittest
from collect_stats import collect, kind_of


class StatisticsTests(unittest.TestCase):
    def test_request_semantics(self):
        base = dict(method='GET', status=200, kind='portable', bytes=100, range='')
        self.assertEqual(kind_of(base), 'portable')
        self.assertEqual(kind_of(dict(base, status=206, range='bytes=0-1023')), 'portable')
        for patch in [dict(method='HEAD'), dict(status=404), dict(bytes=0), dict(kind='asset'),
                      dict(range='bytes=1024-'), dict(range='bytes=-100'), dict(range='bytes=0-20,40-60')]:
            self.assertIsNone(kind_of(dict(base, **patch)))
        self.assertEqual(kind_of(dict(base, kind='page')), 'page')
        self.assertIsNone(kind_of(dict(base, kind='page', status=206)))

    def test_increment_rotation_compression_partial_and_restart(self):
        with tempfile.TemporaryDirectory(prefix='ptb-website-stats-') as tmp:
            root = Path(tmp)
            log = root / 'ptb-website-stats.log'
            state, public = root / 'state.json', root / 'public.json'
            def line(id, kind='page'):
                return (json.dumps(dict(id=id, method='GET', status=200, bytes=200, kind=kind))+'\n').encode()
            log.write_bytes(line('a') + line('b', 'installer'))
            first = collect(root, state, public)
            self.assertEqual(first['pageViews'], 1)
            self.assertEqual(first['downloads']['total'], 1)
            self.assertEqual(collect(root, state, public)['pageViews'], 1)
            log.rename(root / 'ptb-website-stats.log.1')
            log.write_bytes(line('c') + line('d')[:-2])
            self.assertEqual(collect(root, state, public)['pageViews'], 2)
            with log.open('ab') as f: f.write(line('d')[-2:])
            with gzip.open(root / 'ptb-website-stats.log.1.gz', 'wb') as f:
                f.write((root / 'ptb-website-stats.log.1').read_bytes())
            result = collect(root, state, public)
            self.assertEqual(result['pageViews'], 3)
            self.assertEqual(result['downloads']['total'], 1)
            self.assertEqual(result['startedAt'], first['startedAt'])
            self.assertNotIn('offsets', json.loads(public.read_text()))
            state.write_text('{broken', encoding='utf-8')
            with self.assertRaises(ValueError): collect(root, state, public)
            self.assertEqual(json.loads(public.read_text())['pageViews'], 3)


if __name__ == '__main__': unittest.main()
