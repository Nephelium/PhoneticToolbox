"""A denied HTTPS request reaches only the owned loopback proxy."""
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from owned_rejecting_proxy import RejectingProxy


def test_owned_proxy_rejects_connect_without_external_resolution(tmp_path):
    report = tmp_path / 'proxy.json'
    with RejectingProxy(report) as proxy:
        opener = build_opener(ProxyHandler({'https': proxy.url}))
        with pytest.raises(URLError, match='502'):
            opener.open('https://manual.invalid/never-resolved', timeout=2)
        assert proxy.requests == [{'method': 'CONNECT', 'target': 'manual.invalid:443',
                                   'status': 502, 'forwarded': False}]
    assert report.is_file()


def test_real_updater_honors_process_environment_and_keeps_partial_failures(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / 'desktop/src'))
    from ptb_desktop.updates import UpdateService
    with RejectingProxy(tmp_path / 'native-proxy.json') as proxy:
        for name, value in proxy.environment().items():
            monkeypatch.setenv(name, value)
        service = UpdateService(current_version='3.1.2', data_root=tmp_path / 'updates', system_country=lambda: 'CN')
        result = service.check(manual=True)
        assert result['status'] == 'incomplete' and result['candidate'] is None
        assert set(result['sources']) == {'server', 'github'}
        for source in result['sources'].values():
            assert source['code'] == 'NETWORK_ERROR'
            assert source['display'] == {'key': 'native.errors.updates.networkFailure'}
        assert {'www.phonetictoolbox.com:443', 'api.github.com:443'} <= {row['target'] for row in proxy.requests}
        assert all(not row['forwarded'] for row in proxy.requests)
