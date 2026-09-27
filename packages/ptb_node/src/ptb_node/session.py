"""Single-attempt engine behind the closed B/P11 integration gates.

Claims and asset objects must come from B's generated validated models. The
binding supplies remaining durations measured at the response's server time.
This module defines no wire request/response schema.
"""
import hashlib
import os
from pathlib import Path
import stat
import threading
import time
from .config import NodeError
from .lease import Heartbeat, Identity, Lease, boot_clock
from .storage import BudgetWriter
from .transport import download, upload, retry


class Session:
    def __init__(self, config, binding, runtime, store, abort_event):
        self.config, self.binding, self.runtime = config, binding, runtime
        self.store, self.abort_event = store, abort_event
        self.phase, self.node_bytes = 'download', 0
        self.directory = None
        self.lease = None
        self._aborted = threading.Event()
        self.phase_lock = threading.Lock()

    def renew(self, identity):
        with self.phase_lock:
            self.check()
            return self.binding.renew(identity, self.phase, self.node_bytes)

    def transition(self, phase):
        # Even very fast computations must acknowledge download -> compute ->
        # upload in order. A periodic heartbeat alone can skip the compute phase.
        with self.phase_lock:
            self.check()
            started = self.lease.clock()
            identity, duration = self.binding.renew(self.lease.identity, phase, self.node_bytes)
            self.lease.renew(identity, duration, started)
            self.phase = phase

    def abort(self):
        # The P11 implementation must implement bounded, idempotent whole-group
        # abort, and separately verify cleanup before returning from execute.
        self._aborted.set()
        try:
            self.binding.abort_transfer()
        finally:
            self.runtime.abort()

    def check(self):
        if self.abort_event.is_set():
            self.lease.stop()
        self.lease.check()

    def wait(self, seconds):
        end = boot_clock() + seconds
        while boot_clock() < end:
            self.check()
            self.abort_event.wait(min(0.05, end - boot_clock()))

    def run(self, claim, *, request_started, lease_seconds, remaining_seconds):
        # No task execution until local evidence and operation/parameters pass
        # P11's validator. Resources are never inferred from remote argv.
        self.runtime.validate(claim, self.config)
        if claim.memory_bytes > self.config.memory_bytes:
            raise NodeError('memory_budget_unavailable')
        remaining_seconds = min(remaining_seconds, *(item.expires_remaining_seconds for item in claim.inputs))
        identity = Identity(str(claim.attempt_id), claim.generation)
        self.lease = Lease(identity, hard_deadline=request_started + remaining_seconds - 5)
        self.lease.renew(identity, lease_seconds, request_started)
        budget = sum(item.size_bytes for item in claim.inputs) + claim.max_output_bytes
        self.directory = self.store.create(identity, time.time() + remaining_seconds, budget)

        heartbeat_worker = Heartbeat(self.lease, self.renew, self.abort).start()
        try:
            inputs = {}
            for item in claim.inputs:
                self.check()
                path, stream = self.store.open_new(self.directory)
                with stream:
                    writer = BudgetWriter(stream, item.size_bytes, self.config.reserve_disk_bytes)
                    def read(offset, count):
                        self.check()
                        return self.binding.read(identity, item, offset, count)
                    download(writer, size=item.size_bytes, sha256=item.sha256,
                             read_chunk=read, check=self.check, wait=self.wait)
                self.node_bytes += item.size_bytes
                inputs[str(item.id)] = path
            self.transition('compute')
            self.check()
            outputs = self.runtime.execute(claim, inputs, self.directory, self.check)
            self.transition('upload')
            total = 0
            for output in outputs:
                self.check()
                # Adapter-approved local flat files only. Reject links and devices,
                # even if an adapter accidentally returns one.
                path = Path(output.path)
                if path.parent != self.directory:
                    raise NodeError('output_path_invalid')
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, 'rb') as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise NodeError('output_path_invalid')
                    total += info.st_size
                    if total > claim.max_output_bytes:
                        raise NodeError('output_budget_exceeded')
                    digest = hashlib.sha256()
                    while chunk := stream.read(262144):
                        self.check()
                        digest.update(chunk)
                    stream.seek(0)
                    sha = digest.hexdigest()
                    upload_id = retry(lambda: self.binding.reserve(identity, output, info.st_size, sha),
                                      self.check, self.wait)
                    # Always resend from offset zero. B must accept identical chunks
                    # after a lost reservation/append response, per frozen binding.
                    upload(stream, size=info.st_size, sha256=sha,
                           write_chunk=lambda offset, data, block_hash: self.binding.write(
                               identity, upload_id, offset, data, block_hash), check=self.check, wait=self.wait)
                    self.node_bytes += info.st_size
            self.check()
            return retry(lambda: self.binding.complete(identity), self.check, self.wait)
        except Exception as exc:
            # Never serialize exception messages, input labels or scientific params.
            code = 'cancelled' if self.abort_event.is_set() else 'resource_limit'
            if isinstance(exc, NodeError) and str(exc) in ('asset_hash_mismatch', 'asset_length_mismatch'):
                code = 'invalid_input'
            if isinstance(exc, NodeError) and str(exc) == 'network_unavailable':
                code = 'network_error'
            try:
                self.lease.check()
                self.binding.fail(identity, code)
            except Exception:
                pass
            raise
        finally:
            self.lease.stop()
            heartbeat_worker.close()
            self.abort()
            # Do not unlink active science files if P11 cannot confirm termination.
            if not self.runtime.cleaned():
                raise NodeError('process_cleanup_failed')
            self.store.cleanup(self.directory)
