"""Host-owned optional-runtime runner. No MFA imports, no global environment edit."""
import json
import os
from pathlib import Path
import time
import sys
from uuid import uuid4
from .components import atomic_json, default_root, digest, no_links

PINNED_VERSION = '3.3.8'
MEMORY_BYTES = 2_147_483_648
TEMP_BYTES = 512_000_000
TIMEOUT_SECONDS = 180


def resolve_runtime(path):
    root = Path(path).absolute()
    no_links(root)
    if (root / 'env').is_dir():
        root = root / 'env'
    executable = root / ('python.exe' if os.name == 'nt' else 'bin/python')
    if not executable.is_file():
        raise ValueError('m11_runtime_missing')
    records = list((root / 'conda-meta').glob('montreal-forced-aligner-*.json'))
    if len(records) != 1 or json.loads(records[0].read_text(encoding='utf-8'))['version'] != PINNED_VERSION:
        raise ValueError('m11_version_mismatch')
    return root, executable


def fingerprint(root, *, stop=lambda: False):
    root, executable = resolve_runtime(root)
    import hashlib
    h = hashlib.sha256()
    import stat
    paths=[]
    def visit(folder):
        with os.scandir(folder) as entries:
            for entry in entries:
                if stop():raise ValueError('cancelled')
                if entry.name=='__pycache__' or (Path(folder)==root and entry.name=='ptb-component.json'):continue
                info=entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
                    raise ValueError('m11_component_link')
                if stat.S_ISDIR(info.st_mode):visit(entry.path)
                elif stat.S_ISREG(info.st_mode):paths.append(Path(entry.path))
                else:raise ValueError('m11_component_link')
    visit(root)
    # Every code/native/data file is hashed; inspect each directory entry once.
    # Installed package metadata is not treated as proof of binary integrity.
    for p in sorted(paths):
        if stop():raise ValueError('cancelled')
        h.update(p.relative_to(root).as_posix().encode())
        h.update(bytes.fromhex(digest(p)))
    return h.hexdigest()


def run(runtime, workspace, request, *, stop=lambda: False, progress=lambda _: None,
        timeout=TIMEOUT_SECONDS, memory=MEMORY_BYTES, disk=TEMP_BYTES, evidence=None):
    runtime, python = resolve_runtime(runtime)
    root = Path(workspace).absolute()
    no_links(root)
    root.mkdir(parents=True, exist_ok=True)
    request = dict(request, workspace=str(root), runtime=str(runtime))
    atomic_json(root / 'request.json', request)
    child = Path(sys._MEIPASS)/'resources/mfa/child.py' if getattr(sys,'frozen',False) else Path(__file__).with_name('child.py').absolute()
    if not child.is_file():raise ValueError('m11_bootstrap_missing')
    argv = [str(python), '-I', '-B', str(child), str(root / 'request.json')]
    if os.name != 'nt':
        # Linux optional environment has NOT passed P11 systemd + MFA receipt gate.
        # Never silently replace this with an unrestricted Popen/process group.
        raise ValueError('m11_linux_runtime_unverified')
    from ..native.windows import OwnedProcess
    process = None
    started = time.monotonic()
    peak_disk = 0
    stage = None
    try:
        if stop():
            raise ValueError('cancelled')
        process = OwnedProcess(argv, root, memory)
        while True:
            if stop():
                raise ValueError('cancelled')
            if time.monotonic() - started > timeout:
                raise ValueError('m11_timeout')
            size = 0
            for p in root.rglob('*'):
                if p.is_symlink() or getattr(p, 'is_junction', lambda: False)():
                    raise ValueError('m11_attempt_link')
                try:
                    if p.is_file():
                        size += p.stat().st_size
                except FileNotFoundError:
                    pass
            peak_disk = max(peak_disk, size)
            if size > disk:
                raise ValueError('m11_temp_budget')
            status = root / 'status.json'
            if status.is_file():
                value = json.loads(status.read_text(encoding='utf-8'))
                if value['stage'] != stage:
                    stage = value['stage']
                    progress(stage)
            code = process.poll()
            if code is not None:
                if code:
                    raise ValueError('m11_process_crashed')
                response = root / 'response.json'
                if not response.is_file() or response.stat().st_size > 8_000_000:
                    raise ValueError('m11_response_missing')
                result = json.loads(response.read_text(encoding='utf-8'))
                if not result.get('success'):
                    raise ValueError(result.get('error', 'm11_execution_failed'))
                return result
            time.sleep(.1)
    finally:
        if process:
            try:
                if evidence is not None:
                    evidence.update(memory_peak_bytes=process.memory_peak(), temp_peak_sampled_bytes=peak_disk,
                                    elapsed_seconds=time.monotonic()-started, configured_memory_bytes=memory,
                                    configured_temp_bytes=disk, main_pid=process.pid)
            finally:
                process.close()
                if evidence is not None:
                    evidence['group_cleaned'] = process.group_cleaned


def registry_root():
    # Host-only configuration. A task request cannot choose this path.
    return Path(os.environ.get('PTB_M11_COMPONENT_ROOT', str(default_root()))).absolute()


def load_registry():
    path = registry_root() / 'registry.json'
    no_links(path)
    if not path.exists():
        return dict(schema='m11-registry/1', runtimes=[], models=[])
    return json.loads(path.read_text(encoding='utf-8'))


def select(runtime_id, model_id, *, verify_content=True, stop=lambda: False):
    registry = load_registry()
    runtimes = [r for r in registry['runtimes'] if r['id'] == runtime_id]
    models = [m for m in registry['models'] if m['id'] == model_id]
    if len(runtimes) != 1 or len(models) != 1:
        raise ValueError('m11_component_not_registered')
    runtime, model = runtimes[0], models[0]
    if runtime.get('validated') is not True or model.get('validated_runtime') != runtime_id:
        raise ValueError('m11_self_test_required')
    no_links(runtime['receipt'])
    try:
        if digest(runtime['receipt']) != runtime['receipt_sha256']:
            raise ValueError('m11_self_test_required')
        receipt = json.loads(Path(runtime['receipt']).read_text(encoding='utf8'))
    except (OSError,KeyError,json.JSONDecodeError):
        raise ValueError('m11_self_test_required') from None
    if not receipt.get('success') or receipt.get('runtime_fingerprint') != runtime['fingerprint']:
        raise ValueError('m11_self_test_required')
    if verify_content and fingerprint(runtime['path'],stop=stop) != runtime['fingerprint']:
        raise ValueError('m11_runtime_changed')
    for role in ('model', 'dictionary'):
        no_links(model[role])
        try:
            if digest(model[role]) != model[role + '_sha256']:
                raise ValueError('m11_model_changed')
        except OSError:
            raise ValueError('m11_model_missing') from None
    return runtime, model
