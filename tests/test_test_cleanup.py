"""Test cleanup planning against an independent, isolated Git repository."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_test_cleanup import prepare


class TestCleanupPlan(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ptb-cleanup-plan-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        (self.root / '.gitignore').write_text('output/\n', encoding='utf-8')
        self.write('output/validation/case/report.json', {'success': True})
        self.file = self.root / 'output/validation/case/scratch/input.wav'
        self.file.parent.mkdir()
        self.file.write_bytes(b'generated test input')
        self.target = dict(path='output/validation/case/scratch', group='test', reason='Owned fixture',
                           evidence=[dict(path='output/validation/case/report.json', require={'success': True})])

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def plan(self, target=None, **extra):
        return prepare(self.root, dict(schema='ptb-test-cleanup-rules/1',
                                       discover_completed_scratch=False,
                                       targets=[target or self.target], **extra))

    def test_manifest_records_exact_bytes_and_never_deletes(self):
        result = self.plan()
        row = result['targets'][0]['files'][0]
        self.assertEqual(row['bytes'], 20)
        self.assertEqual(row['sha256'], hashlib.sha256(b'generated test input').hexdigest())
        self.assertTrue(self.file.exists())

    def test_tracked_content_is_excluded_even_when_ignored(self):
        subprocess.run(['git', '-C', str(self.root), 'add', '-f', str(self.file)], check=True)
        result = self.plan()
        self.assertFalse(result['targets'])
        self.assertIn('tracked', result['skipped'][0]['reason'])
        if os.name == 'nt':
            alias = dict(self.target, path=self.target['path'].upper())
            result = self.plan(alias)
            self.assertFalse(result['targets'])
            self.assertIn('tracked', result['skipped'][0]['reason'])

    def test_active_scope_and_ancestor_are_protected(self):
        for protected in ('output/validation/case', 'output/validation/case/scratch/input.wav'):
            with self.subTest(protected=protected):
                self.assertFalse(self.plan(protect=[protected])['targets'])

    def test_external_and_drive_relative_paths_are_rejected(self):
        for path in ('../outside', str(self.root), 'C:relative', '/absolute', 'output/validation'):
            with self.subTest(path=path):
                target = dict(self.target, path=path)
                self.assertFalse(self.plan(target)['targets'])

    def test_evidence_must_survive_and_meet_completion_condition(self):
        for proof in (dict(path='output/validation/case/scratch/input.wav'),
                      dict(path='output/validation/case/report.json', require={'success': False})):
            with self.subTest(proof=proof):
                self.assertFalse(self.plan(dict(self.target, evidence=[proof]))['targets'])

    def test_ready_scratch_requires_unchanged_finalizer_inventory(self):
        self.write('output/validation/case/run.json', {'schema': 'ptb-test-run/1', 'state': 'ready'})
        self.write('output/validation/case/scratch-manifest.json',
                   {'schema': 'ptb-test-scratch-files/1', 'files': []})
        rules = dict(schema='ptb-test-cleanup-rules/1', targets=[])
        self.assertFalse(prepare(self.root, rules)['targets'])
        self.write('output/validation/case/scratch-manifest.json',
                   {'schema': 'ptb-test-scratch-files/1', 'files': [
                       dict(path='input.wav', bytes=20, sha256=hashlib.sha256(self.file.read_bytes()).hexdigest())]})
        result = prepare(self.root, rules)
        self.assertEqual(len(result['targets']), 1)
        self.assertIn('output/validation/case/scratch-manifest.json', result['retained_evidence_sha256'])
        self.write('output/validation/case/run.json', {'schema': 'ptb-test-run/1', 'state': 'active'})
        self.assertFalse(prepare(self.root, rules)['targets'])

    def test_link_cannot_adopt_another_directory(self):
        outside = self.root / 'outside'
        outside.mkdir()
        sentinel = outside / 'keep.txt'
        sentinel.write_bytes(b'original must survive')
        link = self.file.parent / 'linked'
        if os.name == 'nt':
            quote = lambda p: str(p).replace("'", "''")
            subprocess.run(['pwsh.exe', '-NoProfile', '-Command',
                            f"New-Item -ItemType Junction -Path '{quote(link)}' -Target '{quote(outside)}' | Out-Null"], check=True)
        else:
            link.symlink_to(outside, target_is_directory=True)
        try:
            result = self.plan()
            self.assertFalse(result['targets'])
            self.assertIn('Linked', result['skipped'][0]['reason'])
            self.assertEqual(sentinel.read_bytes(), b'original must survive')
        finally:
            if os.name == 'nt':
                assert link.is_relative_to(self.root)
                subprocess.run(['pwsh.exe', '-NoProfile', '-Command',
                                f"Remove-Item -LiteralPath '{quote(link)}' -Force"], check=True)
            else:
                link.unlink()


if __name__ == '__main__':
    unittest.main()
