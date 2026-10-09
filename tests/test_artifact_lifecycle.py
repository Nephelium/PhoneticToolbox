"""Small filesystem tests for ownership and evidence preservation; no audio/GUI."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_artifacts import build_workspace
from verification_artifacts import inventory, no_links, verification_run


class ArtifactLifecycleTests(unittest.TestCase):
    def test_manifest_and_cleanup_failure_preserve_evidence_and_external_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            external = root / 'original.wav'
            external.write_bytes(b'original')
            with patch('verification_artifacts.subprocess.run', side_effect=OSError('locked')):
                with verification_run('test', 'case', 'tiny synthetic bytes', root=root) as (out, scratch):
                    (scratch / 'generated.wav').write_bytes(b'generated')
                    (out / 'report.json').write_text('{"passed":true}')
            self.assertEqual(external.read_bytes(), b'original')
            self.assertTrue((out / 'report.json').exists())
            self.assertTrue(scratch.exists())
            files = json.loads((out / 'scratch-manifest.json').read_text())['files']
            self.assertEqual(files, [dict(path='generated.wav', bytes=9, sha256=hashlib.sha256(b'generated').hexdigest())])
            self.assertEqual(json.loads((out / 'cleanup.json').read_text())['status'], 'retained')

    def test_original_failure_survives_cleanup_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('verification_artifacts.subprocess.run', side_effect=OSError('locked')):
                with self.assertRaisesRegex(AssertionError, 'scientific mismatch'):
                    with verification_run('test', 'case', 'failure', root=Path(folder)) as (out, scratch):
                        (scratch / 'partial.wav').write_bytes(b'partial')
                        raise AssertionError('scientific mismatch')
            self.assertEqual(json.loads((out / 'run.json').read_text())['test_outcome'], 'failed')

    def test_traversal_names_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                with verification_run('../manual', 'case', 'invalid', root=Path(folder)):
                    self.fail('invalid name accepted')
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_links_rejected(self):
        with patch.object(Path, 'is_symlink', return_value=True):
            with self.assertRaises(ValueError):
                no_links(Path('linked'))

    def test_build_success_marked(self):
        with tempfile.TemporaryDirectory() as folder:
            with build_workspace(folder, 'build'):
                self.assertEqual(json.loads((Path(folder) / 'artifact-lifecycle.json').read_text())['state'], 'active')
            self.assertEqual(json.loads((Path(folder) / 'artifact-lifecycle.json').read_text())['state'], 'completed')

    def test_interrupted_build_is_identifiable(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(KeyboardInterrupt):
                with build_workspace(folder, 'installer'):
                    raise KeyboardInterrupt()
            marker = json.loads((Path(folder) / 'artifact-lifecycle.json').read_text())
            self.assertEqual((marker['state'], marker['error_type']), ('failed', 'KeyboardInterrupt'))


if __name__ == '__main__':
    unittest.main()
