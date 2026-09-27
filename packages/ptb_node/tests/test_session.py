"""Internal port fixtures, NOT B wire contracts or scientific equivalence tests."""
import hashlib
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as Obj
import unittest

from ptb_node.config import Config, NodeError
from ptb_node.lease import boot_clock
from ptb_node.session import Session
from ptb_node.storage import Attempts


class Binding:
    def __init__(self):
        self.failures, self.complete_calls, self.blocks = [], 0, {}
        self.lost_complete = False
        self.corrupt = False
        self.aborted = False
        self.phases = []

    def renew(self, identity, phase, node_bytes):
        self.phases.append(phase)
        return identity, 60

    def read(self, identity, item, offset, count):
        data = b'bad' if self.corrupt else b'abc'
        return data[offset:offset+count]

    def reserve(self, identity, output, size, sha):
        return 'synthetic-upload'

    def write(self, identity, upload_id, offset, data, block_hash):
        self.blocks[offset] = data
        return offset + len(data)

    def complete(self, identity):
        self.complete_calls += 1
        if self.lost_complete and self.complete_calls == 1:
            raise NodeError('network_unavailable')
        return {'synthetic_receipt': 1}

    def fail(self, identity, code):
        self.failures.append(code)

    def abort_transfer(self):
        self.aborted = True


class Runtime:
    def __init__(self):
        self.started, self.stopped, self.cleanup_ok = False, False, True
        self.invalid = None

    def validate(self, claim, config):
        if self.invalid:
            raise NodeError(self.invalid)

    def execute(self, claim, inputs, directory, check):
        self.started = True
        check()
        target = directory / 'synthetic-output'
        target.write_bytes(next(iter(inputs.values())).read_bytes())
        return [Obj(path=target, key='result', name='synthetic.bin')]

    def abort(self):
        self.stopped = True

    def cleaned(self):
        return self.cleanup_ok


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = Config('https://example.invalid', self.root, self.root / 'credential', reserve_disk_bytes=1)
        self.store = Attempts(self.root, 1024, 1)
        self.binding, self.runtime = Binding(), Runtime()
        self.abort = threading.Event()
        self.session = Session(self.config, self.binding, self.runtime, self.store, self.abort)
        self.claim = Obj(attempt_id='synthetic-attempt', generation=1,
                         inputs=[Obj(id='synthetic-asset', size_bytes=3, sha256=hashlib.sha256(b'abc').hexdigest(), expires_remaining_seconds=120)],
                         max_output_bytes=20, memory_bytes=67108864)

    def tearDown(self):
        self.temp.cleanup()

    def run_session(self):
        return self.session.run(self.claim, request_started=boot_clock(), lease_seconds=60, remaining_seconds=120)

    def test_full_component_flow_and_cleanup(self):
        self.assertEqual(self.run_session(), {'synthetic_receipt': 1})
        self.assertEqual(self.binding.blocks, {0: b'abc'})
        self.assertTrue(self.runtime.stopped)
        self.assertFalse(list(self.store.root.iterdir()))
        self.assertEqual(self.binding.phases, ['compute', 'upload'])

    def test_lost_complete_response_uses_same_attempt(self):
        self.binding.lost_complete = True
        self.session.wait = lambda _: self.session.check()
        self.run_session()
        self.assertEqual(self.binding.complete_calls, 2)
        self.assertEqual(self.claim.generation, 1)

    def test_corrupt_input_never_executes(self):
        self.binding.corrupt = True
        with self.assertRaisesRegex(NodeError, 'asset_hash_mismatch'):
            self.run_session()
        self.assertFalse(self.runtime.started)
        self.assertEqual(self.binding.failures, ['invalid_input'])
        self.assertFalse(list(self.store.root.iterdir()))

    def test_runtime_version_mismatch_stops_before_download(self):
        self.runtime.invalid = 'version_mismatch'
        with self.assertRaisesRegex(NodeError, 'version_mismatch'):
            self.run_session()
        self.assertFalse(list(self.store.root.iterdir()))

    def test_missing_font_stops_before_download(self):
        self.runtime.invalid = 'font_unavailable'
        with self.assertRaisesRegex(NodeError, 'font_unavailable'):
            self.run_session()
        self.assertFalse(self.runtime.started)

    def test_process_cleanup_failure_preserves_residue_and_blocks(self):
        self.runtime.cleanup_ok = False
        with self.assertRaisesRegex(NodeError, 'process_cleanup_failed'):
            self.run_session()
        self.assertTrue(list(self.store.root.iterdir()))

    def test_cancellation_never_publishes(self):
        self.abort.set()
        with self.assertRaisesRegex(NodeError, 'lease_lost'):
            self.run_session()
        self.assertEqual(self.binding.complete_calls, 0)
        self.assertFalse(self.runtime.started)

    def test_over_memory_budget_never_starts(self):
        self.claim.memory_bytes = self.config.memory_bytes + 1
        with self.assertRaisesRegex(NodeError, 'memory_budget_unavailable'):
            self.run_session()
        self.assertFalse(self.runtime.started)

    def test_input_expiry_before_task_deadline(self):
        self.claim.inputs[0].expires_remaining_seconds = 1
        with self.assertRaisesRegex(NodeError, 'lease_lost'):
            self.run_session()
        self.assertFalse(list(self.store.root.iterdir()))


if __name__ == '__main__':
    unittest.main()
