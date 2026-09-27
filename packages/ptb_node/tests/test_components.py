"""Public synthetic inputs; these are client tests, never server transaction evidence."""
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from ptb_node.config import Config, NodeError, credential, origin, save_credential
from ptb_node.lease import Heartbeat, Identity, Lease
from ptb_node.runtime import P11Runtime
from ptb_node.storage import Attempts, BudgetWriter, Instance, private_dir
from ptb_node.transport import download, upload, backoff
from ptb_node.service import Service, control


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.resume = 0.0
        self.identity = Identity('attempt-a', 1)
        self.lease = Lease(self.identity, clock=lambda: self.now, resume_clock=lambda: self.resume)
        self.lease.renew(self.identity, 60, 10)

    def test_delayed_grant_anchored_to_request(self):
        self.now = 20
        self.lease.renew(self.identity, 60, 15)
        self.assertEqual(self.lease.deadline, 70)

    def test_expired_grant_never_revives(self):
        self.now = 65
        with self.assertRaisesRegex(NodeError, 'lease_lost'):
            self.lease.renew(self.identity, 60, 65)

    def test_old_result_cannot_change_generation(self):
        with self.assertRaisesRegex(NodeError, 'lease_lost'):
            self.lease.renew(Identity('attempt-a', 2), 60, 10)
        self.assertEqual(self.lease.identity.generation, 1)

    def test_suspension_before_expiry_still_invalidates(self):
        self.now = 12
        self.resume = 2
        with self.assertRaisesRegex(NodeError, 'lease_lost'):
            self.lease.check()

    def test_invalid_duration(self):
        with self.assertRaisesRegex(NodeError, 'lease_invalid'):
            self.lease.renew(self.identity, float('nan'), 10)

    def test_input_expiry_caps_lease(self):
        lease = Lease(self.identity, clock=lambda: self.now, hard_deadline=20)
        lease.renew(self.identity, 60, 10)
        self.assertEqual(lease.deadline, 20)

    def test_download_or_compute_block_cannot_block_watchdog(self):
        aborted = threading.Event()
        release = threading.Event()
        def blocked(_):
            release.wait(2)
            return self.identity, 60
        heartbeat = Heartbeat(self.lease, blocked, aborted.set, interval=0.01).start()
        time.sleep(0.03)
        self.now = 65
        self.assertTrue(aborted.wait(0.5))
        release.set()
        heartbeat.close()

    def test_revocation_stops_without_waiting_expiry(self):
        aborted = threading.Event()
        def reject(_):
            raise NodeError('credential_rejected')
        heartbeat = Heartbeat(self.lease, reject, aborted.set, interval=0.01).start()
        self.assertTrue(aborted.wait(0.5))
        heartbeat.close()

    def test_transient_network_failure_then_renew(self):
        calls = []
        def flaky(identity):
            calls.append(1)
            if len(calls) == 1:
                raise NodeError('network_unavailable')
            return identity, 70
        heartbeat = Heartbeat(self.lease, flaky, lambda: None, interval=0.01).start()
        end = time.monotonic() + 1
        while len(calls) < 2 and time.monotonic() < end:
            time.sleep(0.01)
        heartbeat.close()
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(self.lease.deadline, 75)


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.data = bytes(range(256)) * 4096
        self.sha = hashlib.sha256(self.data).hexdigest()

    def test_download_retries_same_offset(self):
        seen = []
        def read(offset, count):
            seen.append(offset)
            if len(seen) == 1:
                raise NodeError('network_unavailable')
            return self.data[offset:offset+count]
        output = io.BytesIO()
        download(output, size=len(self.data), sha256=self.sha, read_chunk=read, check=lambda: None, wait=lambda _: None)
        self.assertEqual(output.getvalue(), self.data)
        self.assertEqual(seen[:2], [0, 0])

    def test_wrong_hash_rejected(self):
        with self.assertRaisesRegex(NodeError, 'asset_hash_mismatch'):
            download(io.BytesIO(), size=2, sha256='0'*64, read_chunk=lambda *_: b'ab', check=lambda: None, wait=lambda _: None)

    def test_truncation_rejected(self):
        with self.assertRaisesRegex(NodeError, 'asset_length_mismatch'):
            download(io.BytesIO(), size=2, sha256='0'*64, read_chunk=lambda *_: b'a', check=lambda: None, wait=lambda _: None)

    def test_upload_identical_retry_and_bounded_blocks(self):
        seen = []
        def write(offset, data, sha):
            seen.append((offset, data, sha))
            if len(seen) == 1:
                raise NodeError('network_unavailable')
            self.assertLessEqual(len(data), 262144)
            return offset + len(data)
        upload(io.BytesIO(self.data), size=len(self.data), sha256=self.sha, write_chunk=write, check=lambda: None, wait=lambda _: None)
        self.assertEqual(seen[0], seen[1])

    def test_upload_offset_rejected(self):
        with self.assertRaisesRegex(NodeError, 'upload_offset_mismatch'):
            upload(io.BytesIO(b'ab'), size=2, sha256='0'*64, write_chunk=lambda *_: 3, check=lambda: None, wait=lambda _: None)

    def test_network_retry_is_bounded(self):
        calls = []
        def fail(*_):
            calls.append(1)
            raise NodeError('network_unavailable')
        with self.assertRaisesRegex(NodeError, 'network_unavailable'):
            download(io.BytesIO(), size=2, sha256='0'*64, read_chunk=fail, check=lambda: None, wait=lambda _: None)
        self.assertEqual(len(calls), 5)

    def test_revoked_lease_prevents_next_chunk(self):
        calls = []
        def check():
            if calls:
                raise NodeError('lease_lost')
        def read(*_):
            calls.append(1)
            return b'ab'
        with self.assertRaisesRegex(NodeError, 'lease_lost'):
            download(io.BytesIO(), size=2, sha256='0'*64, read_chunk=read, check=check, wait=lambda _: None)
        self.assertEqual(len(calls), 1)

    def test_backoff_cap(self):
        self.assertLessEqual(backoff(10000), 30)


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = Config('https://example.invalid', self.root / 'state', self.root / 'credential')

    def tearDown(self):
        self.temp.cleanup()

    def test_origin_validation(self):
        for value in ('http://example.org', 'https://user:pass@example.org', 'https://example.org/x', 'https://example.org?key=secret'):
            with self.subTest(value=value), self.assertRaises(NodeError):
                origin(value)

    def test_config_rejects_parallel_and_injected_command(self):
        data = dict(schema='ptb-node-config/1', server_origin='https://example.org',
                    state_dir=str(self.cfg.state_dir), credential_file=str(self.cfg.credential_file), parallelism=2)
        file = self.root / 'config.json'
        file.write_text(json.dumps(data))
        with self.assertRaises(NodeError):
            Config.load(file)
        data['parallelism'] = 1
        data['command'] = 'arbitrary shell'
        file.write_text(json.dumps(data))
        with self.assertRaises(NodeError):
            Config.load(file)

    def test_work_hours(self):
        cfg = Config('https://example.org', self.cfg.state_dir, self.cfg.credential_file, work_hours=(20, 24))
        self.assertFalse(cfg.in_hours(datetime(2026, 9, 27, 12)))
        self.assertTrue(cfg.in_hours(datetime(2026, 9, 27, 21)))

    def test_credential_mode_and_symlink(self):
        path = self.cfg.credential_file
        path.write_text('synthetic-token')
        path.chmod(0o644)
        with self.assertRaisesRegex(NodeError, 'credential_permissions'):
            credential(path)
        path.chmod(0o600)
        self.assertEqual(credential(path), 'synthetic-token')
        link = self.root / 'link'
        link.symlink_to(path)
        with self.assertRaises(NodeError):
            credential(link)

    def test_credential_provisioning_is_private_and_no_overwrite(self):
        save_credential(self.cfg.credential_file, 'synthetic-only')
        self.assertEqual(self.cfg.credential_file.stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(NodeError, 'credential_write_failed'):
            save_credential(self.cfg.credential_file, 'replacement')
        self.assertEqual(credential(self.cfg.credential_file), 'synthetic-only')

    def test_state_symlink_rejected(self):
        target = private_dir(self.root / 'real')
        self.cfg.state_dir.symlink_to(target)
        with self.assertRaises(NodeError):
            private_dir(self.cfg.state_dir)

    def test_single_instance(self):
        with Instance(self.cfg.state_dir):
            with self.assertRaisesRegex(NodeError, 'already_running'):
                with Instance(self.cfg.state_dir):
                    pass

    def test_disk_shortage(self):
        store = Attempts(self.root, 100, 1)
        with patch('ptb_node.storage.shutil.disk_usage') as disk:
            disk.return_value.free = 10
            with self.assertRaisesRegex(NodeError, 'disk_budget_unavailable'):
                store.create(Identity('a', 1), 100, 50)

    def test_owned_recovery_preserves_symlink_target(self):
        store = Attempts(self.root, 100, 1)
        attempt = store.create(Identity('a', 1), 100, 50)
        other = self.root / 'unrelated'
        other.write_text('keep')
        (attempt / 'escape').symlink_to(other)
        store.recover()
        self.assertEqual(other.read_text(), 'keep')
        self.assertFalse(attempt.exists())

    def test_unknown_residue_blocks_recovery(self):
        store = Attempts(self.root, 100, 1)
        (store.root / 'unrelated').mkdir()
        with self.assertRaisesRegex(NodeError, 'cleanup_not_owned'):
            store.recover()

    def test_cleanup_failure_blocks_new_attempt(self):
        store = Attempts(self.root, 100, 1)
        attempt = store.create(Identity('a', 1), 100, 50)
        with patch('ptb_node.storage.shutil.rmtree', side_effect=PermissionError) as remove:
            remove.avoids_symlink_attacks = True
            with self.assertRaisesRegex(NodeError, 'cleanup_failed'):
                store.cleanup(attempt)
        with self.assertRaisesRegex(NodeError, 'residual_cleanup_required'):
            store.create(Identity('b', 2), 100, 50)

    def test_output_budget(self):
        with (self.root / 'data').open('wb') as file:
            writer = BudgetWriter(file, 2)
            with self.assertRaisesRegex(NodeError, 'disk_budget_exceeded'):
                writer.write(b'abc')
        self.assertEqual((self.root / 'data').stat().st_size, 0)

    def test_p11_gate_cannot_be_bypassed(self):
        self.assertEqual(P11Runtime().capabilities()['modules'], [])
        with self.assertRaisesRegex(NodeError, 'p11_trusted_worker_unavailable'):
            P11Runtime().execute({'operation': 'shell', 'path': '/bin/sh'})

    def test_pause_is_distinct_from_abort(self):
        service = Service(self.cfg)
        service.control('pause')
        self.assertFalse(service.abort.is_set())
        service.control('resume')
        self.assertFalse(service.paused)
        service.control('abort')
        self.assertTrue(service.abort.is_set())

    def test_health_reconnect_and_goodbye(self):
        class Fake:
            calls = 0
            gone = False
            def health(self, _):
                self.calls += 1
                if self.calls == 1:
                    raise NodeError('network_unavailable')
            def goodbye(self):
                self.gone = True
        binding = Fake()
        service = Service(self.cfg, binding=binding)
        with patch('ptb_node.service.backoff', return_value=0.01):
            thread = threading.Thread(target=service._work)
            thread.start()
            deadline = time.monotonic() + 1
            while binding.calls < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            service.control('stop')
            thread.join(2)
        self.assertGreaterEqual(binding.calls, 2)
        self.assertTrue(binding.gone)

    def test_real_service_control_and_unrelated_process(self):
        cfg_file = self.root / 'config.json'
        cfg_file.write_text(json.dumps({'schema': 'ptb-node-config/1', 'server_origin': 'https://example.invalid',
                                       'state_dir': str(self.cfg.state_dir), 'credential_file': str(self.cfg.credential_file)}))
        args = [sys.executable, '-B', '-m', 'ptb_node', '--config', str(cfg_file)]
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(15)'])
        node = subprocess.Popen(args + ['start'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 3
            while not (self.cfg.state_dir / 'control.sock').exists() and node.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(control(self.cfg, 'pause')['paused'])
            self.assertFalse(control(self.cfg, 'resume')['paused'])
            second = subprocess.run(args + ['start'], capture_output=True, timeout=3)
            self.assertEqual(second.returncode, 2)
            self.assertIn(b'already_running', second.stderr)
            control(self.cfg, 'stop')
            node.wait(timeout=4)
            self.assertEqual(node.returncode, 0)
            self.assertIsNone(unrelated.poll())
            self.assertFalse((self.cfg.state_dir / 'control.sock').exists())
        finally:
            if node.poll() is None:
                node.terminate()
            node.communicate(timeout=5)
            unrelated.terminate()
            unrelated.wait(timeout=3)


if __name__ == '__main__':
    unittest.main()
