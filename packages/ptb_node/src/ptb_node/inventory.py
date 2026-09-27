"""Read-only inventory: no GPU/model discovery scan or service changes."""
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def read(path):
    try:
        return Path(path).read_text(encoding='utf-8').strip()
    except OSError:
        return None


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=3, stdin=subprocess.DEVNULL)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def inspect(root):
    memory = {}
    for line in (read('/proc/meminfo') or '').splitlines():
        key, _, value = line.partition(':')
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal'):
            memory[key + '_bytes'] = int(value.strip().split()[0]) * 1024
    versions = {}
    for name in ('phonetic-core', 'ptb-api', 'numpy', 'scipy', 'praat-parselmouth', 'matplotlib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    systemctl, systemd_run = shutil.which('systemctl'), shutil.which('systemd-run')
    manager = command([systemctl, '--user', 'show', '--property=Version', '--value']) if systemctl else None
    fonts = command(['fc-list', '--format=%{family}\n']) if shutil.which('fc-list') else None
    root = Path(root)
    while not root.exists():
        root = root.parent
    return {
        'schema': 'ptb-node-inventory/1', 'os_release': read('/etc/os-release'),
        'platform': platform.system(), 'architecture': platform.machine(),
        'kernel': platform.release(), 'python': platform.python_version(),
        'python_executable': sys.executable, 'cpu_count': os.cpu_count(),
        'memory': memory, 'disk_free_bytes': shutil.disk_usage(root).free,
        'pid1': read('/proc/1/comm'), 'cgroup_membership': read('/proc/self/cgroup'),
        'cgroup_controllers': read('/sys/fs/cgroup/cgroup.controllers'),
        'cgroup_root_writable': os.access('/sys/fs/cgroup/cgroup.procs', os.W_OK),
        'systemd_run_available': bool(systemd_run), 'user_systemd_available': bool(manager),
        'hard_limits': 'unverified' if systemd_run and manager else 'unavailable',
        'versions': versions, 'fonts': sorted(set((fonts or '').splitlines())),
        'gpu': 'not_advertised', 'modules': [], 'network': 'not_probed',
    }
