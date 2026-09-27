"""P11-ENV read-only inventory. No installation, service start, DDL or cleanup.

Run with the target's Linux Python. Exit 2 means inventory/acceptance is incomplete,
not that Linux was verified. --audit-root also works on Windows and never executes
legacy validators. Reports may contain local paths: keep them in ignored output/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener


PACKAGES = ('ptb-api', 'phonetic-core', 'fastapi', 'pydantic', 'uvicorn',
            'psycopg', 'numpy', 'scipy', 'praat-parselmouth', 'matplotlib', 'pandas')
AUDIT_PATTERNS = {
    'windows_path_or_executable': r'[A-Za-z]:[/\\]|python\.exe|node\.exe|chrome\.exe|Scripts[/\\]',
    'windows_process_or_pipe': r'native\.windows|InputPipe|OwnedProcess|CREATE_NO_WINDOW|WinDLL|win32|taskkill',
    'browser_or_qt': r'chrome|chromium|playwright|PyQt|QApplication|QtWebEngine|node',
    'database_or_ddl': r'sqlite|postgres|psycopg|migration|executescript|\b(?:CREATE|ALTER|DROP|TRUNCATE)\b',
    'deletion_or_recovery': r'\.unlink\(|rmtree\(|\bDELETE\b|\.cleanup\(|\.recover\(|release_scratch',
    'service_or_process': r'Popen|subprocess|LocalService|server\.main|ptb_api\.server',
    'private_or_baseline_input': r'private|natural|牧歌|1[–-]4\.wav|baseline',
}


def audit_scripts(root: Path) -> list[dict]:
    """Static indicators only: a clean scan is never permission to run a script."""
    records = []
    for path in sorted((root / 'scripts').glob('verify_m*_*.py')):
        raw = path.read_bytes()
        lines = raw.decode('utf-8-sig').splitlines()
        matches = {key: [i for i, line in enumerate(lines, 1) if re.search(pattern, line, re.I)]
                   for key, pattern in AUDIT_PATTERNS.items()}
        records.append({'path': path.relative_to(root).as_posix(),
                        'sha256': hashlib.sha256(raw).hexdigest(),
                        'indicators': {k: v for k, v in matches.items() if v},
                        'execution_authorized_by_scan': False})
    return records


def read_text(path: Path, maximum: int = 65536) -> str | None:
    try:
        with path.open('r', encoding='utf-8') as stream:
            value = stream.read(maximum + 1)
        return value if len(value) <= maximum else None
    except (OSError, UnicodeError):
        return None


def command(argv: list[str]) -> dict:
    if not shutil.which(argv[0]):
        return {'status': 'missing'}
    try:
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                                timeout=5, check=False, encoding='utf-8', errors='replace')
        # Never return stderr, environment, process command lines, or DB settings.
        return {'status': 'observed' if result.returncode == 0 else 'failed',
                'returncode': result.returncode, 'stdout': result.stdout[:32768],
                'truncated': len(result.stdout) > 32768}
    except subprocess.TimeoutExpired:
        return {'status': 'timeout'}
    except OSError:
        return {'status': 'unavailable'}


def cgroup_inventory(root: Path, membership: str) -> dict:
    """Read the caller's v2 group, without creating groups or testing writes."""
    entry = next((line[3:] for line in membership.splitlines() if line.startswith('0::')), None)
    if entry is None:
        return {'status': 'v2_membership_unavailable', 'enforcement_verified': False}
    relative = PurePosixPath(entry)
    if not relative.is_absolute() or '..' in relative.parts:
        return {'status': 'invalid_membership', 'enforcement_verified': False}
    base = root.resolve()
    target = (base / entry.lstrip('/')).resolve()
    if not target.is_relative_to(base):
        return {'status': 'invalid_membership', 'enforcement_verified': False}
    names = ('cgroup.controllers', 'cgroup.subtree_control', 'cgroup.events',
             'memory.current', 'memory.peak', 'memory.max', 'memory.swap.max',
             'memory.events', 'cpu.max', 'pids.max')
    values = {name: read_text(target / name) for name in names}
    status = ('observed' if values['cgroup.events'] is not None else
              'observed_root' if target == base and values['cgroup.controllers'] is not None else
              'mount_unresolved')
    return {'status': status,
            'mount_assumption': '/sys/fs/cgroup; namespaces/nonstandard mounts require review',
            'values': values, 'kill_interface_present': (target / 'cgroup.kill').exists(),
            'directory_writable_hint': os.access(target, os.W_OK),
            'enforcement_verified': False}


def package_inventory() -> dict:
    result = {}
    for name in PACKAGES:
        try:
            dist = metadata.distribution(name)
            # Paths are local evidence; do not expose direct_url or index credentials.
            result[name] = {'version': dist.version, 'location': str(dist.locate_file('')),
                            'wheel_metadata': dist.read_text('WHEEL') is not None}
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def loopback_origin(url: str) -> str:
    parsed = urlsplit(url)
    if (parsed.scheme != 'http' or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in ('', '/')
            or not parsed.hostname):
        raise ValueError('Expected a plain loopback HTTP origin without credentials')
    if not ipaddress.ip_address(parsed.hostname).is_loopback:
        raise ValueError('Only literal loopback addresses are supported')
    if parsed.port is None:
        raise ValueError('Explicit loopback port required')
    return url.rstrip('/')


def http_inventory(origin: str) -> dict:
    """Read an already running, explicitly selected host. Do not start a server."""
    origin = loopback_origin(origin)
    opener = build_opener(ProxyHandler({}), NoRedirect())
    result = {}
    for route in ('/api/v1/health', '/api/v1/capabilities', '/server/'):
        try:
            with opener.open(origin + route, timeout=5) as response:
                raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ValueError('response_budget')
                media = response.headers.get_content_type()
                valid = False
                if route == '/api/v1/health' and media == 'application/json':
                    value = json.loads(raw)
                    valid = isinstance(value, dict) and value.get('status') == 'ok'
                elif route == '/api/v1/capabilities' and media == 'application/json':
                    value = json.loads(raw)
                    valid = isinstance(value, dict) and isinstance(value.get('algorithms'), list)
                elif route == '/server/' and media == 'text/html':
                    valid = '<html' in raw.decode('utf-8').lower()
                result[route] = {'status': response.status, 'content_type': media,
                                 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                                 'shape_valid': valid}
        except (OSError, ValueError) as exc:
            result[route] = {'status': 'failed', 'error_type': type(exc).__name__}
    # Do not save response content: capability strings can contain private paths.
    return {'routes': result, 'font_rendering_verified': False,
            'scientific_capability_verified': False}


def inventory(project: Path, base_url: str | None = None) -> dict:
    report = {'schema': 'p11-environment/1', 'platform': platform.system(),
              'machine': platform.machine(), 'python_version': platform.python_version(),
              'python_executable': sys.executable, 'project_path': str(project.resolve()),
              'linux_execution': sys.platform == 'linux' and not sys.executable.lower().endswith('.exe'),
              'acceptance': 'incomplete', 'blockers': [],
              'scientific_modules_enabled_by_probe': [], 'packages': package_inventory()}
    report['project_markers'] = {name: (project / name).is_file() for name in
                                 ('backend/pyproject.toml', 'packages/phonetic_core/pyproject.toml',
                                  'frontend/dist/index.html')}
    if not report['linux_execution']:
        report['blockers'].append('requires_native_linux_python')
        return report
    report['environment_kind'] = 'wsl' if 'microsoft' in platform.release().lower() else 'linux_host_or_container'
    report['os_release'] = platform.freedesktop_os_release()
    report['cpu_count'] = os.cpu_count()
    report['cpu_affinity_count'] = len(os.sched_getaffinity(0))
    report['meminfo'] = {line.split(':')[0]: line.split(':', 1)[1].strip()
                         for line in (read_text(Path('/proc/meminfo')) or '').splitlines()
                         if line.split(':')[0] in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree')}
    report['disk'] = dict(zip(('total', 'used', 'free'), shutil.disk_usage(project)))
    report['project_mount'] = command(['findmnt', '-n', '-o', 'FSTYPE,TARGET', '-T', str(project.resolve())])
    report['cgroup'] = cgroup_inventory(Path('/sys/fs/cgroup'), read_text(Path('/proc/self/cgroup')) or '')
    report['commands'] = {name: command(argv) for name, argv in {
        'systemd_version': ['systemctl', '--version'],
        'systemd_state': ['systemctl', 'is-system-running'],
        'systemd_user_state': ['systemctl', '--user', 'is-system-running'],
        'postgres_client': ['psql', '--version'],
        'postgres_binary': ['postgres', '--version'],
        'pg_config': ['pg_config', '--version'],
        'listeners': ['ss', '-H', '-ltn'],
        'ipa_font_match': ['fc-match', '-f', '%{family}\n%{file}\n', 'Doulos SIL'],
        'chinese_font_match': ['fc-match', '-f', '%{family}\n%{file}\n', 'sans:lang=zh-cn'],
    }.items()}
    if sys.version_info[:2] != (3, 11):
        report['blockers'].append('project_requires_python_3_11')
    for name in ('ptb-api', 'phonetic-core'):
        if not report['packages'][name]:
            report['blockers'].append('missing_installed_' + name)
    report['blockers'].extend(['linux_lock_not_verified', 'process_group_enforcement_not_verified',
                               'chinese_and_ipa_rendering_not_verified'])
    if base_url:
        report['http'] = http_inventory(base_url)
        if not all(v.get('status') == 200 and v.get('shape_valid') for v in report['http']['routes'].values()):
            report['blockers'].append('api_or_static_http_check_failed')
    else:
        report['blockers'].append('api_and_static_http_not_checked')
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--audit-root', type=Path, help='Only scan legacy source; never execute it')
    parser.add_argument('--base-url', help='Optional existing loopback HTTP origin; no credentials')
    parser.add_argument('--output', type=Path, help='New JSON file; refuses overwrite, parent must exist')
    args = parser.parse_args(argv)
    if args.base_url:
        try:
            loopback_origin(args.base_url)
        except ValueError as exc:
            parser.error(str(exc))
    if args.audit_root:
        if not (args.audit_root / 'scripts').is_dir():
            parser.error('audit root must contain scripts/')
        report = {'schema': 'p11-script-audit/1', 'scripts': audit_scripts(args.audit_root),
                  'linux_execution_verified': False}
        code = 0
    else:
        if not args.project_root.is_dir():
            parser.error('project root must exist')
        report = inventory(args.project_root, args.base_url)
        code = 2  # Readiness needs separate actual lifecycle, wheel and visual tests.
    payload = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(payload)
    else:
        print(payload, end='')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
