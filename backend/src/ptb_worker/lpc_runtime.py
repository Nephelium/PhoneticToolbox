"""Reuse the verified project MKL runtime with an independent fixed LPC entry."""
from pathlib import Path
from .egg_runtime import command as compatible_command, fingerprint
from .acoustic_errors import AcousticFailure


def command(request, pipe):
    try:
        argv = compatible_command(request, pipe)
    except AcousticFailure as exc:
        raise AcousticFailure(exc.code.replace('egg_', 'lpc_')) from None
    argv[3] = str(Path(__file__).with_name('lpc_bootstrap.py'))
    return argv
