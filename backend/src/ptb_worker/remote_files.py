"""Transaction port for the pending P07 bridge, NOT a second storage policy.

Implementations must own Storage -> scheduling lock order and persist file/quota
changes with the supplied transaction. A separate connection in publish is unsafe.
No production implementation is enabled while A owns the shared file pipeline.
"""
from typing import Protocol


class RemoteFiles(Protocol):
    def validate_inputs(self, tx, job) -> list[dict]:
        """Recheck active owner/project, policy, hash, input TTL; return bounded manifest."""

    def read(self, tx, job, asset_id: str, offset: int, size: int) -> bytes: ...
    def reserve(self, tx, job, upload_id: str, name: str, size: int):
        """Reserve in P07 quota before disk writes, including failed old attempts."""

    def append(self, tx, job, upload_id: str, offset: int, data: bytes): ...
    def seal(self, tx, job, upload_id: str, expected_hash: str): ...
    def publish(self, tx, job, uploads: list[dict]) -> dict:
        """Atomically apply P07 TTL/M08 policy and quota; never update jobs itself."""

