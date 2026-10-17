"""Focused regression checks for counting, resume requests and log rotation."""
from pathlib import Path
import gzip
import json
import tempfile
import unittest
from collect_stats import collect, kind_of, compact_history


class StatisticsTests(unittest.TestCase):
    def test_sparse_star_samples_survive_compaction(self):
        points=[[120,1,0,0,33],[125,2,0,0,None],[180,2,1,0,None]]
        self.assertEqual(compact_history(points,180),points)
        self.assertEqual(compact_history(points,180+3*86400),[points[0],points[-1]])

    def test_web_usage_is_new_workspace_only(self):
        row=dict(method='POST',status=200,kind='web',webUse='1')
        self.assertEqual(kind_of(row),'web')
        for patch in [dict(method='GET'),dict(status=401),dict(webUse='0'),dict(webUse='')]:
            self.assertIsNone(kind_of(dict(row,**patch)))

    def test_history_backfill_and_github_failure_are_truthful(self):
        with tempfile.TemporaryDirectory(prefix='ptb-stats-history-') as tmp:
            root=Path(tmp);state=root/'state.json';public=root/'public.json'
            lines=[dict(id='one',time='2026-10-12T10:23:40+00:00',method='GET',status=200,kind='page',bytes=5),
                   dict(id='two',time='2026-10-12T11:25:40+00:00',method='GET',status=200,kind='installer',bytes=5)]
            (root/'ptb-website-stats.log').write_text(''.join(json.dumps(x)+'\n' for x in lines))
            value=collect(root,state,public,fetch_stars=lambda:33)
            points=json.loads((root/'site-stats-history.json').read_text())['points']
            self.assertEqual(points[0][1:5],[1,0,None,None])
            self.assertEqual(points[-1][1:5],[1,1,0,33])
            self.assertIsNone(points[-1][5])
            saved=json.loads(state.read_text());saved['githubAttempt']=0;state.write_text(json.dumps(saved))
            def failure():raise OSError('synthetic offline')
            second=collect(root,state,public,fetch_stars=failure)
            self.assertEqual(second['github'],value['github'])
            self.assertEqual(second['pageViews'],1)

    def test_request_semantics(self):
        base = dict(method='GET', status=200, kind='portable', bytes=100, range='')
        self.assertEqual(kind_of(base), 'portable')
        self.assertEqual(kind_of(dict(base, status=206, range='bytes=0-1023')), 'portable')
        self.assertEqual(kind_of(dict(base, kind='macos')), 'macos')
        self.assertEqual(kind_of(dict(base, kind='macos',status=206,range='bytes=0-1023')), 'macos')
        self.assertIsNone(kind_of(dict(base, kind='macos',status=206,range='bytes=1024-')))
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

    def test_macos_backfill_preserves_history_and_deduplicates_live_requests(self):
        with tempfile.TemporaryDirectory(prefix='ptb-macos-stats-') as tmp:
            root=Path(tmp);state=root/'state.json';public=root/'public.json'
            before=1792221600
            original=[[before-3600,12,20,3,33],[before+300,13,20,3,None]]
            state.write_text(json.dumps(dict(schema=1,startedAt='2026-10-12T10:23:30+00:00',
                counts=dict(page=13,portable=8,installer=12,web=3),offsets={},invalidLines=0,history=original)))
            def access(time,status=200,agent='Mozilla/5.0',method='GET',size=123):
                return f'198.51.100.1 - - [17/Oct/2026:{time} +0800] "{method} /releases/macos-arm64/3.0.3/PhoneticToolbox-3.0.3-macos14-arm64-preview-r6.zip HTTP/1.1" {status} {size} "-" "{agent}"\n'
            log=access('18:01:00')+access('18:02:00',206)+access('18:03:00',agent='PTB-Website-Verify')+access('18:04:00',method='HEAD')+access('18:05:00',size=0)+access('18:06:00')
            (root/'access.log.1').write_text(log)
            with gzip.open(root/'access.log.1.gz','wt') as stream:stream.write(log)
            anonymous=dict(id='mac-live',time='2026-10-17T18:06:00+08:00',method='GET',kind='macos',status=200,bytes=123,range='')
            (root/'ptb-website-stats.log').write_text(json.dumps(anonymous)+'\n')
            result=collect(root,state,public,backfill_access_logs=True)
            self.assertEqual(result['downloads'],dict(total=22,windows=20,macos=2,portable=8,installer=12,macosStartedAt='2026-10-17T10:01:00+00:00'))
            saved=json.loads(state.read_text())
            self.assertEqual(saved['macosBackfill']['imported'],1)
            self.assertEqual(saved['macosBackfill']['skippedUnverifiableRanges'],1)
            by_time={p[0]:p for p in saved['history']}
            for p in original:self.assertEqual(by_time[p[0]][:5],p)
            self.assertIsNone(by_time[original[0][0]][5])
            self.assertEqual(collect(root,state,public,backfill_access_logs=True)['downloads']['macos'],2)
            self.assertNotIn('198.51.100.1',public.read_text())
            self.assertNotIn('Mozilla',state.read_text())

    def test_macos_live_rotation_and_compaction(self):
        with tempfile.TemporaryDirectory(prefix='ptb-macos-live-') as tmp:
            root=Path(tmp);state=root/'state.json';public=root/'public.json'
            row=dict(id='a',time='2026-10-17T18:01:00+08:00',method='GET',kind='macos',status=206,bytes=123,range='bytes=0-1023')
            log=root/'ptb-website-stats.log';log.write_text(json.dumps(row)+'\n')
            self.assertEqual(collect(root,state,public,backfill_access_logs=True)['downloads']['macos'],1)
            log.rename(root/'ptb-website-stats.log.1')
            log.write_text(json.dumps(dict(row,id='b',range='bytes=1024-'))+'\n')
            self.assertEqual(collect(root,state,public)['downloads']['macos'],1)
            points=[[120,1,2,3,33,None],[125,None,None,None,None,1],[180,2,3,3,None,1]]
            self.assertIn(points[0],compact_history(points,180+3*86400))
            self.assertIn(points[-1],compact_history(points,180+3*86400))


if __name__ == '__main__': unittest.main()
