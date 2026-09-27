"""Host-owned, hash-bound Linux runtime profile; no client-controlled commands."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

from ..io.limits import FormatError

ENTRIES = {'m06': 'ptb_worker.m06_child', 'm08': 'ptb_worker.m08_child', 'lpc': 'ptb_worker.lpc_child', 'egg': 'ptb_worker.egg_child',
           'acoustic': 'ptb_worker.science_child', 'segment': 'ptb_worker.segment_child',
           'fonts': 'ptb_worker.font_preflight', 'spectrogram': 'ptb_worker.spectrogram_preview'}


def load_profile(path=None):
    try:
        return _load_profile(path)
    except FormatError:
        raise
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        raise FormatError('linux_runtime_mismatch') from None


def _load_profile(path=None):
    path = Path(path or os.environ.get('PTB_LINUX_RUNTIME_PROFILE', ''))
    if not path.is_absolute() or not path.is_file() or path.stat().st_size > 131072:
        raise FormatError('linux_runtime_unavailable')
    profile = json.loads(path.read_text('utf-8'))
    if profile.get('schema') != 'p11-runtime/1':
        raise FormatError('linux_runtime_mismatch')
    for key in ('python', 'cache'):
        if not Path(profile[key]).is_absolute():
            raise FormatError('linux_runtime_mismatch')
    if not Path(profile['python']).is_file() or not Path(profile['cache']).is_dir():
        raise FormatError('linux_runtime_unavailable')
    for folder in profile['sys_paths']:
        if not Path(folder).is_absolute() or not Path(folder).is_dir():
            raise FormatError('linux_runtime_mismatch')
    hashes = profile['hashes']
    bootstrap = str(Path(__file__).parents[1]/'linux_bootstrap.py')
    if not hashes or profile['python'] not in hashes or bootstrap not in hashes:
        raise FormatError('linux_runtime_mismatch')
    for filename, expected in hashes.items():
        item = Path(filename)
        if not item.is_absolute() or hashlib.sha256(item.read_bytes()).hexdigest() != expected:
            raise FormatError('linux_runtime_mismatch')
    return path, profile


def command(entry, request):
    if entry not in ENTRIES:
        raise ValueError('Unsupported fixed Linux entry')
    path, profile = load_profile()
    return [profile['python'], '-I', '-B', str(Path(__file__).parents[1]/'linux_bootstrap.py'),
            str(path), entry, str(request)]


def fingerprint():
    path, profile = load_profile()
    actual = {name: metadata.version(name) for name in profile['versions']}
    if actual != profile['versions'] or Path(sys.executable).resolve() != Path(profile['python']).resolve():
        raise FormatError('linux_runtime_mismatch')
    import numpy as np
    import scipy
    return dict(platform='linux', versions=actual,
                profile_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                blas=scipy.__config__.CONFIG['Build Dependencies']['blas']['name'],
                numpy_blas=np.__config__.CONFIG['Build Dependencies']['blas']['name'],
                equivalence_policy='p11-practical/1')


def run(entry, request, cwd, limits, *, stop=lambda: False, on_started=None, evidence=None, on_chunk=None):
    from ..resource_profiles import selected_profile
    from .posix import run_bounded
    # Cloud policy must not silently limit an explicitly configured local desktop.
    limits = selected_profile().limits(limits)
    return run_bounded(command(entry, request), b'', cwd, limits, stop=stop,
                       on_started=on_started, evidence=evidence, on_chunk=on_chunk)
