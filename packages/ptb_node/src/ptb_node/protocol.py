"""Internal orchestration ports, NOT a wire schema or guessed B endpoints.

An eventual fixed binding must validate B-generated models, scope and version,
map durations from server grants, and never manufacture a new generation.
"""
from typing import Protocol
from .config import NodeError


class Binding(Protocol):
    def health(self, capabilities: dict) -> None: ...
    def poll(self): ...
    def goodbye(self) -> None: ...


class UnavailableBinding:
    def health(self, capabilities):
        raise NodeError('protocol_unavailable')

    def poll(self):
        raise NodeError('protocol_unavailable')

    def goodbye(self):
        pass  # No endpoint is invented, and no network request is sent.
