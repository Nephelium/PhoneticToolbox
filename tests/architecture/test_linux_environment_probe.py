"""P11-ENV regression guards; Windows unit tests do not imply Linux acceptance."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[2] / 'scripts/verify_linux_environment.py'
SPEC = importlib.util.spec_from_file_location('p11_probe', SOURCE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_windows_cannot_be_reported_as_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(probe, 'command', lambda *_: pytest.fail('must not run Linux commands'))
    report = probe.inventory(tmp_path, 'http://127.0.0.1:9999')
    assert report['linux_execution'] is False
    assert report['acceptance'] == 'incomplete'
    assert report['blockers'] == ['requires_native_linux_python']


@pytest.mark.parametrize('url', [
    'http://example.com:80', 'http://127.0.0.1:80/?token=x',
    'http://user:secret@127.0.0.1:80', 'http://127.0.0.1',
    'https://127.0.0.1:443', 'http://127.0.0.1:80/api',
    'http://127.0.0.1:80/#secret', 'http://localhost:80',
])
def test_probe_rejects_remote_or_credential_urls(url):
    with pytest.raises(ValueError):
        probe.loopback_origin(url)


def test_loopback_ipv4_and_ipv6():
    assert probe.loopback_origin('http://127.0.0.1:8123/') == 'http://127.0.0.1:8123'
    assert probe.loopback_origin('http://[::1]:8123') == 'http://[::1]:8123'


def test_cgroup_namespace_escape_rejected(tmp_path):
    assert probe.cgroup_inventory(tmp_path, '0::/../../secret')['status'] == 'invalid_membership'


def test_cgroup_observation_never_means_enforcement(tmp_path):
    group = tmp_path / 'test'
    group.mkdir()
    (group / 'cgroup.events').write_text('populated 1\n', encoding='utf-8')
    (group / 'memory.max').write_text('1073741824\n', encoding='utf-8')
    report = probe.cgroup_inventory(tmp_path, '0::/test')
    assert report['status'] == 'observed'
    assert report['values']['memory.max'] == '1073741824\n'
    assert report['values']['memory.peak'] is None
    assert report['enforcement_verified'] is False


def test_root_cgroup_without_per_group_counters_is_still_observed(tmp_path):
    (tmp_path / 'cgroup.controllers').write_text('cpu memory pids\n', encoding='utf-8')
    report = probe.cgroup_inventory(tmp_path, '0::/')
    assert report['status'] == 'observed_root'
    assert report['values']['memory.current'] is None
    assert report['enforcement_verified'] is False


def test_legacy_scan_is_complete_and_does_not_execute(tmp_path):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    (scripts / 'verify_m01_danger.py').write_text(
        'raise RuntimeError("must not execute")\n# CREATE TABLE; rmtree(x); node.exe\n', encoding='utf-8')
    (scripts / 'verify_m02_clean.py').write_text('pass\n', encoding='utf-8')
    records = probe.audit_scripts(tmp_path)
    assert len(records) == 2
    assert records[0]['indicators']['database_or_ddl'] == [2]
    assert records[0]['indicators']['deletion_or_recovery'] == [2]
    assert all(not row['execution_authorized_by_scan'] for row in records)
    assert records[1]['indicators'] == {}


def test_output_refuses_to_replace_evidence(tmp_path):
    (tmp_path / 'scripts').mkdir()
    output = tmp_path / 'result.json'
    output.write_text('prior evidence', encoding='utf-8')
    with pytest.raises(FileExistsError):
        probe.main(['--audit-root', str(tmp_path), '--output', str(output)])
    assert output.read_text('utf-8') == 'prior evidence'


def test_cli_windows_incomplete_exit_and_json(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'platform', 'win32')
    output = tmp_path / 'report.json'
    assert probe.main(['--project-root', str(tmp_path), '--output', str(output)]) == 2
    assert json.loads(output.read_text('utf-8'))['scientific_modules_enabled_by_probe'] == []


def test_http_probe_checks_content_and_bounds(monkeypatch):
    class Headers:
        def __init__(self, media):
            self.media = media

        def get_content_type(self):
            return self.media

    class Response:
        status = 200

        def __init__(self, raw, media):
            self.raw, self.headers = raw, Headers(media)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, maximum):
            return self.raw[:maximum]

    class Opener:
        def open(self, url, timeout):
            assert timeout == 5
            if url.endswith('/health'):
                return Response(b'{"status":"broken"}', 'application/json')
            if url.endswith('/capabilities'):
                return Response(b'{"algorithms":[]}', 'application/json')
            return Response(b'<' * 2_000_001, 'text/html')

    monkeypatch.setattr(probe, 'build_opener', lambda *args: Opener())
    report = probe.http_inventory('http://127.0.0.1:8123')
    assert report['routes']['/api/v1/health']['shape_valid'] is False
    assert report['routes']['/api/v1/capabilities']['shape_valid'] is True
    assert report['routes']['/server/']['status'] == 'failed'
    assert report['scientific_capability_verified'] is False


def test_http_redirects_are_not_followed():
    assert probe.NoRedirect().redirect_request(None, None, 302, '', {}, 'http://example.com') is None


def test_inventory_does_not_accept_windows_executable_under_linux_label(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(sys, 'executable', '/mnt/d/runtime/python.exe')
    report = probe.inventory(tmp_path)
    assert report['linux_execution'] is False


def test_bounded_proc_read(tmp_path):
    path = tmp_path / 'proc-fixture'
    path.write_text('x' * 17, encoding='utf-8')
    assert probe.read_text(path, maximum=16) is None
