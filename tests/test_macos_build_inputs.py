"""Mac packaging contracts runnable on Windows; no native approval is simulated."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from macos_build_inputs import DATA_FILES, DATA_TREES, SOURCE_TREES, app_version, freeze_inputs, verify_snapshot
from macos_validation_identity import runtime_hashes, source_hashes
from register_macos_validation import CHECKS, VERSIONS, validate_reports
from restore_macos_environment import npm_command


class MacBuildContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='ptb-macos-contract-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'source'
        self.root.mkdir()
        for base in SOURCE_TREES + DATA_TREES:
            (self.root / base).mkdir(parents=True, exist_ok=True)
        for path in DATA_FILES:
            self.put(path, '{}')
        self.put('release/version.json', json.dumps({'app_version': '3.0.5'}))
        self.put('backend/src/test_module.py', 'value=1\n')
        self.put('scripts/macos_science_entry.py', 'pass\n')
        self.put('scripts/macos_validation_identity.py', 'pass\n')
        self.put('scripts/verify_macos_runtime.py', 'pass\n')
        self.put('resources/vocal_tract/native/darwin-arm64/test.dylib', 'native-fixture')

    def put(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf8')

    def test_explicit_roots_keep_reaper_and_exclude_private_data_and_mfa(self):
        for name in ('frontend/dist/geolinguistics/catalog.json', 'local-data/private-scan.pdf',
                     'resources/research/private-scan.pdf', 'resources/research/mfa/model.zip',
                     'contracts/internal.md', 'backend/src/__pycache__/stale.pyc'):
            self.put(name, 'not-for-distribution')
        self.put('frontend/dist/index.html', 'application')
        snapshot = self.root.parent / 'snapshot'
        manifest = freeze_inputs(self.root, snapshot)
        names = {row['path'] for row in manifest['files']}
        self.assertEqual(manifest['app_version'], '3.0.5')
        self.assertIn('resources/research/reaper', names)
        self.assertIn('frontend/dist/index.html', names)
        self.assertFalse(any('private-scan' in p or '/mfa/' in p or 'geolinguistics' in p for p in names))
        self.assertNotIn('contracts/internal.md', names)
        self.assertNotIn('backend/src/__pycache__/stale.pyc', names)
        verify_snapshot(snapshot, manifest)
        (snapshot / 'backend/src/test_module.py').write_text('changed', encoding='utf8')
        with self.assertRaisesRegex(RuntimeError, 'Frozen build input changed'):
            verify_snapshot(snapshot, manifest)

    def reports(self):
        return ({'success': True, 'checks': list(CHECKS), 'runtime': {'versions': VERSIONS},
                 'source_sha256': runtime_hashes(self.root)},
                {'success': True, 'source_sha256': source_hashes(self.root)})

    def test_current_source_reports_are_accepted(self):
        runtime, numeric = self.reports()
        self.assertEqual(validate_reports(runtime, numeric, self.root), runtime_hashes(self.root))

    def test_old_success_without_identity_is_rejected(self):
        runtime, numeric = self.reports()
        del runtime['source_sha256']
        with self.assertRaisesRegex(RuntimeError, 'Runtime report'):
            validate_reports(runtime, numeric, self.root)

    def test_changed_source_or_native_invalidates_success(self):
        runtime, numeric = self.reports()
        self.put('resources/research/reaper', 'replacement-native')
        with self.assertRaisesRegex(RuntimeError, 'Runtime report'):
            validate_reports(runtime, numeric, self.root)
        runtime, numeric = self.reports()
        self.put('backend/src/test_module.py', 'value=2\n')
        runtime['source_sha256'] = runtime_hashes(self.root)
        with self.assertRaisesRegex(RuntimeError, 'Numerical report'):
            validate_reports(runtime, numeric, self.root)

    def test_npm_network_is_explicit_and_task_cached(self):
        cache = self.root.parent / 'cache/npm'
        self.assertIn('--offline', npm_command('npm', cache))
        command = npm_command('npm', cache, allow_network=True)
        self.assertNotIn('--offline', command)
        self.assertIn(str(cache), command)
        self.assertEqual(command[1], 'ci')

    def test_version_comes_from_current_release(self):
        self.assertEqual(app_version(self.root), '3.0.5')
        self.put('release/version.json', '{"app_version":"3.0.6"}')
        self.assertEqual(app_version(self.root), '3.0.6')


if __name__ == '__main__':
    unittest.main()
