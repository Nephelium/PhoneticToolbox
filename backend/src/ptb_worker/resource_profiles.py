"""Host-selected execution policy. These limits never change scientific inputs."""
from dataclasses import dataclass, replace
import os
import sys

from .io.limits import FormatError


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    shared_admission: bool
    memory_bytes: int | None
    queue_limit: int = 128
    queue_seconds: float = 300.0

    def limits(self, limits):
        if self.memory_bytes is None:
            return limits
        return replace(limits, process_bytes=min(limits.process_bytes, self.memory_bytes))


PROFILES = {
    'server-small': ResourceProfile('server-small', True, 1_073_741_824),
    'desktop-local': ResourceProfile('desktop-local', False, None),
    # P06-REMOTE has no node registration/lease/result protocol yet.
    'trusted-worker': ResourceProfile('trusted-worker', False, None),
}


def selected_profile():
    name = os.environ.get('PTB_RESOURCE_PROFILE', 'server-small' if sys.platform == 'linux' else 'desktop-local')
    if name not in PROFILES:
        raise FormatError('resource_profile_invalid')
    if name == 'trusted-worker':
        raise FormatError('trusted_worker_unavailable')
    if name == 'server-small' and sys.platform != 'linux':
        raise FormatError('server_resource_platform_unavailable')
    return PROFILES[name]


def require_server_profile():
    if selected_profile().name != 'server-small':
        raise FormatError('server_resource_profile_required')
