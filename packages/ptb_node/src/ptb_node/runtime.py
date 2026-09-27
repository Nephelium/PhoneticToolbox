"""Fail-closed join to the P11-owned adapter. No alternative science runner."""
from .config import NodeError


class P11Runtime:
    def capabilities(self):
        # P11 currently deliberately rejects trusted-worker. The node must not
        # relabel itself server-small or desktop-local to bypass that gate.
        return {'modules': [], 'reason': 'p11_trusted_worker_unavailable'}

    def validate(self, attempt, config):
        raise NodeError('p11_trusted_worker_unavailable')

    def execute(self, attempt, *args):
        raise NodeError('p11_trusted_worker_unavailable')

    def abort(self):
        pass  # No process can be started while the gate above is closed.

    def cleaned(self):
        return True
