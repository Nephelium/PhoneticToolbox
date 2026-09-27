"""Fixed application stdio entry selection for the Linux process boundary.

Windows production callers keep their existing suspended/handshake Job adapters.
Scientific named-pipe protocols are not silently converted into stdio protocols.
"""
import sys

from ..io.limits import FormatError
from ..process_entry import command

STDIO_MODULES = frozenset({'ptb_worker.parameter_preview'})


def run_fixed_module(module, payload, cwd, limits, *, stop=lambda: False, evidence=None):
    if module not in STDIO_MODULES:
        raise ValueError('Unsupported fixed stdio entry')
    if sys.platform != 'linux':
        raise FormatError('linux_process_boundary_unavailable')
    from .posix import run_bounded
    return run_bounded(command(module), payload, cwd, limits, stop=stop, evidence=evidence)
