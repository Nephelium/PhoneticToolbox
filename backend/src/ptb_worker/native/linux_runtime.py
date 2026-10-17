"""Host-owned, hash-bound Linux runtime profile; no client-controlled commands."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import threading

from ..io.limits import FormatError

ENTRIES = {'m07':'ptb_worker.m07_child','m14':'ptb_worker.m14_child','m06': 'ptb_worker.m06_child', 'm08': 'ptb_worker.m08_child', 'lpc': 'ptb_worker.lpc_child', 'egg': 'ptb_worker.egg_child',
           'acoustic': 'ptb_worker.science_child', 'segment': 'ptb_worker.segment_child',
           'fonts': 'ptb_worker.font_preflight', 'spectrogram': 'ptb_worker.spectrogram_preview', 'recording':'ptb_worker.recording_child', 'recording_preview':'ptb_worker.recording_preview', 'papers':'ptb_documents.web_child',
           'spec2wav':'ptb_worker.spec2wav_child', 'spec2wav_preview':'ptb_worker.spec2wav_preview',
           'egg_interactive':'ptb_worker.egg_interactive_child','parameter_preview':'ptb_worker.parameter_preview'}


_digest_cache = {}
_digest_lock = threading.Lock()
_profile_lock = threading.RLock()


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode)


def _verified_digest(path, expected=None):
    """Reuse a digest only while the trusted Linux file's identity is unchanged.

    Every admission still stats every declared file. ctime detects an in-place
    edit even when size and mtime are restored; inode detects atomic replacement.
    The lock coalesces concurrent cold requests without retaining file contents.
    """
    with _digest_lock:
        before = _signature(path.stat())
        cached = _digest_cache.get(str(path))
        if cached is not None and cached[0] == before and (expected is None or cached[1]==expected):
            return cached[1]
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            if _signature(os.fstat(stream.fileno())) != before:
                raise FormatError('linux_runtime_mismatch')
            for chunk in iter(lambda: stream.read(1_048_576), b''):
                digest.update(chunk)
            if _signature(os.fstat(stream.fileno())) != before:
                raise FormatError('linux_runtime_mismatch')
        if _signature(path.stat()) != before:
            raise FormatError('linux_runtime_mismatch')
        result = digest.hexdigest()
        if len(_digest_cache) >= 4096:
            _digest_cache.clear()
        _digest_cache[str(path)] = (before, result)
        return result


def load_profile(path=None):
    try:
        from .digest_cache import shared_digests
        with _profile_lock, shared_digests(_digest_cache):
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
        if not item.is_absolute() or _verified_digest(item,expected) != expected:
            raise FormatError('linux_runtime_mismatch')
    return path, profile


def command(entry, request):
    if entry not in ENTRIES:
        raise ValueError('Unsupported fixed Linux entry')
    path, profile = load_profile()
    argv = [profile['python'], '-I', '-B', str(Path(__file__).parents[1]/'linux_bootstrap.py'),
            str(path), entry, str(request)]
    # Headless Qt may need project-local GL loader libraries. Scope their search
    # path to the paper child; scientific workers keep their original environment.
    if entry == 'papers' and profile.get('paper_libraries'):
        directory = Path(profile['paper_libraries'])
        if not directory.is_absolute() or not directory.is_dir() or directory.resolve() != directory:
            raise FormatError('linux_runtime_mismatch')
        return ['/usr/bin/env', 'LD_LIBRARY_PATH=' + str(directory), *argv]
    return argv


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
