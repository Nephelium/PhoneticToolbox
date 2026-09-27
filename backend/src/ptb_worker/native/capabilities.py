"""Conservative local runtime eligibility; no DB object is execution evidence.

Linux scientific eligibility stays closed while its scientific baseline and
production call-site integration are incomplete. Remote-node readiness belongs
to P06-REMOTE and is never inferred here. No heavy scientific imports in the API.
"""
import hashlib
import importlib.metadata as metadata
from pathlib import Path
import sys
import os
import json
import subprocess


def platform_name():
    return sys.platform


def m01_capability(store):
    if not getattr(store, 'batches', None):
        return False, 'acoustic_batches_not_configured'
    if platform_name() == 'linux':
        ready, reasons = linux_capabilities(store)
        return ('acoustic_analysis' in ready, None if 'acoustic_analysis' in ready else reasons[0])
    if platform_name() != 'win32':
        return False, 'linux_scientific_runtime_unverified' if platform_name() == 'linux' else 'scientific_platform_unverified'
    binary = getattr(getattr(store, 'files', None), 'reaper_binary', None)
    if not binary:
        return False, 'registered_reaper_unavailable'
    try:
        from .reaper import REAPER_SHA256
        from ..io.scratch import no_links
        path = Path(binary)
        if not path.is_absolute() or not path.is_file():
            return False, 'registered_reaper_unavailable'
        no_links(path)
        if path.stat().st_size > 16_000_000 or hashlib.sha256(path.read_bytes()).hexdigest() != REAPER_SHA256:
            return False, 'registered_reaper_mismatch'
    except (OSError, ValueError, ImportError):
        return False, 'registered_reaper_unavailable'
    expected = {'phonetic-core': '3.0.0a1', 'numpy': '2.2.6', 'scipy': '1.16.3',
                'praat-parselmouth': '0.4.7', 'pandas': '2.3.3'}
    try:
        if any(metadata.version(name) != version for name, version in expected.items()):
            return False, 'm01_runtime_version_mismatch'
    except metadata.PackageNotFoundError:
        return False, 'm01_runtime_packages_unavailable'
    return True, None


def linux_capabilities(store):
    """Require host-selected receipts bound to current runtime and real reports."""
    unavailable = ([], ['linux_scientific_runtime_unverified'])
    if not getattr(store, 'files', None) or not getattr(store, 'batches', None):
        return unavailable
    try:
        from ..resource_profiles import selected_profile
        selected_profile()  # Unknown profiles and unimplemented remote nodes fail closed.
        from .linux_runtime import load_profile
        path, profile = load_profile()
        receipt_path = Path(os.environ.get('PTB_LINUX_VALIDATION_RECEIPT', ''))
        if not receipt_path.is_absolute() or receipt_path.stat().st_size > 32768:
            return unavailable
        receipt = json.loads(receipt_path.read_text('utf-8'))
        if receipt['profile_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            return unavailable
        if getattr(store,'max_running',None)!=1:
            return [], ['linux_scheduler_single_slot_required']
        from .posix import _show
        if _show('ptb-p11-capability-probe.service').get('LoadState') != 'not-found':
            return [], ['linux_process_boundary_unavailable']
        ready=[]
        for operation, phase in [('lpc_analysis','lpc'),('egg_analysis','egg'),('acoustic_analysis','acoustic')]:
            entry=receipt.get('operations',{}).get(operation)
            if not entry:continue
            report_path=Path(entry['report'])
            if not report_path.is_absolute() or report_path.stat().st_size>1000000:continue
            raw=report_path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=entry['sha256']:continue
            report=json.loads(raw)
            if report.get('success') is not True or report.get('phase')!=phase or report.get('profile_sha256')!=receipt['profile_sha256']:continue
            if operation=='acoustic_analysis' and str(getattr(store.files,'reaper_binary',None))!=profile.get('reaper_binary'):continue
            ready.append(operation)
        return ready, ([] if ready else unavailable[1])
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError, subprocess.SubprocessError):
        return unavailable
