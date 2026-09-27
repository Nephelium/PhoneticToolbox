"""Lease duration comes from B's binding; wall-clock lease_until is not trusted locally."""
from dataclasses import dataclass
import math
import threading
import time
from .config import NodeError


def boot_clock():
    # Linux CLOCK_BOOTTIME includes suspend, unlike CLOCK_MONOTONIC.
    return time.clock_gettime(time.CLOCK_BOOTTIME)


@dataclass(frozen=True)
class Identity:
    attempt_id: str
    generation: int


class Lease:
    def __init__(self, identity, *, clock=boot_clock, margin=5.0, hard_deadline=float('inf'), resume_clock=None):
        self.identity = identity
        self.clock = clock
        self.margin = margin
        self.hard_deadline = hard_deadline
        self.deadline = 0.0
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.resume_clock = resume_clock or (lambda: boot_clock() - time.monotonic())
        self.resume_baseline = self.resume_clock()

    def _resumed(self):
        return abs(self.resume_clock() - self.resume_baseline) > 1.0

    def renew(self, identity, granted_seconds, request_started):
        # Anchor at request start, not response arrival; delayed packets never extend a lease.
        with self.lock:
            now = self.clock()
            if (identity != self.identity or self.stopped.is_set() or self._resumed()
                    or (self.deadline and now >= self.deadline)):
                self.stopped.set()
                raise NodeError('lease_lost')
            if (not math.isfinite(granted_seconds) or granted_seconds <= self.margin
                    or not math.isfinite(request_started) or request_started > now):
                self.stopped.set()
                raise NodeError('lease_invalid')
            self.deadline = min(request_started + granted_seconds - self.margin, self.hard_deadline)
            if now >= self.deadline:
                self.stopped.set()
                raise NodeError('lease_lost')

    def check(self):
        if self.stopped.is_set() or self.clock() >= self.deadline or self._resumed():
            self.stopped.set()
            raise NodeError('lease_lost')

    def stop(self):
        self.stopped.set()


class Heartbeat:
    """Heartbeat and watchdog are separate from computation AND blocking transport."""
    def __init__(self, lease, renew, abort, interval=1.0):
        self.lease, self.renew, self.abort, self.interval = lease, renew, abort, interval
        self.closed = threading.Event()
        self.threads = []

    def start(self):
        def watch():
            while not self.closed.wait(0.05):
                try:
                    self.lease.check()
                except NodeError:
                    self.abort()
                    return

        def beat():
            while not self.closed.wait(self.interval):
                started = self.lease.clock()
                try:
                    self.lease.check()
                    identity, duration = self.renew(self.lease.identity)
                    self.lease.renew(identity, duration, started)
                except NodeError as exc:
                    if str(exc) != 'network_unavailable':
                        self.lease.stop()
                        self.abort()
                        return
                except Exception:
                    self.lease.stop()
                    self.abort()
                    return

        self.threads = [threading.Thread(target=f, daemon=True) for f in (watch, beat)]
        for thread in self.threads:
            thread.start()
        return self

    def close(self):
        self.closed.set()
        for thread in self.threads:
            thread.join(timeout=0.2)
