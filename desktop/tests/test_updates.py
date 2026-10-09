"""SemVer, source failures, prompt cadence and verified native download boundaries."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
import pytest

from ptb_desktop.updates import (UpdateService, UpdateError, SemVer, HttpsTransport, _Redirects,
                                _safe_url, SERVER_MANIFEST, GITHUB_RELEASES, REGION_URL)

BODY = b'owned update fixture bytes'
SHA = hashlib.sha256(BODY).hexdigest()
SERVER_PACKAGE = 'https://www.phonetictoolbox.com/releases/windows-x64/preview/PhoneticToolbox-windows-x64-portable.zip'
GITHUB_PACKAGE = 'https://github.com/Nephelium/PhoneticToolbox/releases/download/v3.0.0-preview.2/PhoneticToolbox-windows-x64-portable.zip'


class Response(BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {}


class Transport:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.lock = threading.Lock()

    def open(self, url, *, purpose, timeout):
        _safe_url(url, purpose)
        with self.lock:
            self.calls.append((url, purpose, timeout))
        result = self.responses[url]
        if isinstance(result, Exception):
            raise result
        if callable(result):
            return result()
        if isinstance(result, tuple):
            return Response(*result)
        if isinstance(result, (dict, list)):
            result = json.dumps(result).encode()
        return Response(result)


def manifest(version='3.0.0-preview.2', kind='portable', source='server'):
    return {'schemaVersion': 'ptb-release/1', 'version': version, 'channel': 'preview' if SemVer(version).pre else 'stable',
            'notes': '中文说明，不含本机路径', 'publishedAt': '2026-10-05T00:00:00Z',
            'packages': [{'kind': kind, 'platform': 'windows-x64', 'name': 'PhoneticToolbox-windows-x64-portable.zip' if kind == 'portable' else 'PhoneticToolbox-windows-x64-installer.exe',
                          'url': SERVER_PACKAGE if source == 'server' else GITHUB_PACKAGE, 'size': len(BODY), 'sha256': SHA}]}


def github_release(version='3.0.0-preview.2', digest=True):
    return {'tag_name': f'v{version}', 'prerelease': bool(SemVer(version).pre), 'draft': False, 'body': 'release',
            'assets': [{'name': 'PhoneticToolbox-windows-x64-portable.zip', 'browser_download_url': GITHUB_PACKAGE,
                        'size': len(BODY), 'digest': f'sha256:{SHA}' if digest else None}]}


def make_service(tmp_path, *, server=None, github=None, country='CN', now=None, **kwargs):
    responses = {REGION_URL: f'ip=DO-NOT-STORE\nloc={country}\n'.encode(),
                 SERVER_MANIFEST: server if server is not None else manifest(),
                 SERVER_MANIFEST.replace('/preview/', '/stable/'): server if server is not None else manifest('3.0.0'),
                 f'{GITHUB_RELEASES}?per_page=100&page=1': github if github is not None else [],
                 SERVER_PACKAGE: (BODY, {'Content-Length': str(len(BODY))}),
                 GITHUB_PACKAGE: (BODY, {'Content-Length': str(len(BODY))})}
    transport = Transport(responses)
    service = UpdateService(data_root=tmp_path, transport=transport, clock=(lambda: now[0]) if now else (lambda: 1000000), **kwargs)
    return service, transport


@pytest.mark.parametrize('value', ['0.0.0', '3.0.0-preview.1', '3.0.0-preview.10', '3.0.0-alpha.0', '3.0.0+001', '3.0.0-a-b.1', 'v3.0.0', '3.0.0a1'])
def test_valid_versions(value):
    assert SemVer(value).value


@pytest.mark.parametrize('value', ['01.0.0', '1.00.0', '1.0.01', '1.0', '1.0.0-01', '1.0.0-x.01', '1.0.0-', '1.0.0+', '1.0.0-ä', '１.0.0', None, '3.0.0a01'])
def test_invalid_versions(value):
    with pytest.raises(UpdateError):
        SemVer(value)


def test_complete_semver_precedence_and_build_metadata():
    values = ['1.0.0-alpha', '1.0.0-alpha.1', '1.0.0-alpha.beta', '1.0.0-beta', '1.0.0-beta.2', '1.0.0-beta.11', '1.0.0-rc.1', '1.0.0', '1.0.1']
    assert all(SemVer(a) < SemVer(b) for a, b in zip(values, values[1:]))
    assert SemVer('3.0.0-preview.10') > SemVer('3.0.0-preview.2')
    assert SemVer('3.0.0+build.1') == SemVer('3.0.0+different')
    assert SemVer('3.0.0a1').value == '3.0.0-alpha.1'


@pytest.mark.parametrize('url,purpose', [
    ('http://www.phonetictoolbox.com/releases/a.zip', 'server'),
    ('https://user:secret@www.phonetictoolbox.com/releases/a.zip', 'server'),
    ('https://www.phonetictoolbox.com:444/releases/a.zip', 'server'),
    ('https://www.phonetictoolbox.com/releases/%2e%2e/private', 'server'),
    ('https://www.phonetictoolbox.com/private.zip', 'server'),
    ('https://attacker.test/a.zip', 'download'),
    ('https://github.com/other/other/releases/a.zip', 'download'),
    ('https://api.github.com/repos/other/other/releases', 'github'),
    ('file:///C:/test.exe', 'download'),
    ('https://www.cloudflare.com/other', 'region'),
])
def test_network_allowlist(url, purpose):
    with pytest.raises(UpdateError):
        _safe_url(url, purpose)


def test_redirect_checked_before_request():
    with pytest.raises(UpdateError):
        _Redirects('download').redirect_request(None, None, 302, '', {}, 'https://attacker.test/package.zip')


def test_region_only_loc_and_os_fallback(tmp_path):
    service, transport = make_service(tmp_path, system_country=lambda: 'US')
    assert service.region()['preferredSource'] == 'server'
    service.check(manual=True)
    assert 'DO-NOT-STORE' not in (tmp_path/'state.json').read_text()
    assert 'country' not in json.loads((tmp_path/'state.json').read_text())
    transport.responses[REGION_URL] = TimeoutError('secret body ip=x')
    assert service.region() == {'country': 'US', 'source': 'system', 'label': '系统地区线索', 'preferredSource': 'github'}


def test_manual_preference_skips_region_and_persists(tmp_path):
    service, transport = make_service(tmp_path)
    service.set_preferences({'source': 'github', 'channel': 'stable', 'autoCheck': False})
    assert service.check()['reason'] == 'disabled'
    result = service.check(manual=True)
    assert result['preferredSource'] == 'github' and result['region']['source'] == 'manual'
    assert not any(url == REGION_URL for url, _, _ in transport.calls)
    restarted, _ = make_service(tmp_path)
    assert restarted.preferences()['source'] == 'github'
    assert restarted.preferences()['autoCheck'] is False


def test_highest_valid_semver_across_sources_and_preview(tmp_path):
    service, _ = make_service(tmp_path, github=[github_release('3.0.0-preview.10'), github_release('3.0.0-preview.9'), {'tag_name':'v-invalid'}])
    result = service.check(manual=True)
    assert result['candidate']['version'] == '3.0.0-preview.10'
    assert result['candidate']['source'] == 'github'
    assert result['sources']['server']['status'] == 'checked'


def test_stable_excludes_all_prereleases_and_preview_includes_final(tmp_path):
    service, _ = make_service(tmp_path, github=[github_release('9.0.0-preview.1'), github_release('3.1.0')])
    result = service.check(manual=True, channel='stable')
    assert result['candidate']['version'] == '3.1.0'
    assert result['channel'] == 'stable'
    result = service.check(manual=True, channel='preview')
    assert result['candidate']['version'] == '9.0.0-preview.1'


def test_server_404_and_github_empty_is_unknown(tmp_path):
    missing = HTTPError(SERVER_MANIFEST, 404, 'not found', None, None)
    service, _ = make_service(tmp_path, server=missing)
    result = service.check(manual=True)
    assert result['status'] == 'incomplete'
    assert result['sources']['server']['code'] == 'HTTP_ERROR'
    assert result['sources']['github']['status'] == 'no-releases'


def test_two_source_failures_do_not_claim_no_update_or_leak_error(tmp_path):
    service, _ = make_service(tmp_path, server=TimeoutError('private token'), github=OSError('C:/private/path'))
    result = service.check(manual=True)
    assert result['status'] == 'incomplete' and result['candidate'] is None
    assert set(result['sources']) == {'server', 'github'}
    assert all(source['status'] == 'error' for source in result['sources'].values())
    assert not any(word in json.dumps(result) for word in ('token', 'private/path'))


def test_github_404_does_not_block_server_update(tmp_path):
    service, _ = make_service(tmp_path, github=HTTPError(GITHUB_RELEASES, 404, 'not found', None, None))
    result = service.check(manual=True)
    assert result['status'] == 'available' and result['candidate']['source'] == 'server'
    assert result['sources']['github']['status'] == 'error'


def test_missing_checksum_does_not_claim_up_to_date(tmp_path):
    service, _ = make_service(tmp_path, server=manifest('2.0.0'), github=[github_release('3.0.0-preview.3', digest=False)])
    result = service.check(manual=True)
    assert result['status'] == 'incomplete'
    assert result['sources']['github']['code'] == 'CHECKSUM_MISSING'


def test_every_process_startup_checks_and_once_per_day_prompt(tmp_path):
    now=[1000000]
    service, transport = make_service(tmp_path, now=now)
    result = service.check()
    assert result['shouldPrompt']
    service.acknowledge_notice(result['candidate']['id'])
    call_count = len(transport.calls)
    assert service.check()['status'] == 'deferred' and len(transport.calls) == call_count
    restarted, restarted_transport = make_service(tmp_path, now=now)
    assert restarted.check()['shouldPrompt'] is False
    assert any(purpose=='server' for _,purpose,_ in restarted_transport.calls)
    assert any(purpose=='github' for _,purpose,_ in restarted_transport.calls)
    assert restarted.check(manual=True)['shouldPrompt']
    now[0] += 6*3600
    restarted, _ = make_service(tmp_path, now=now)
    assert restarted.check()['shouldPrompt'] is False
    now[0] += 18*3600
    restarted, _ = make_service(tmp_path, now=now)
    assert restarted.check()['shouldPrompt'] is True


def test_notice_not_suppressed_until_actual_ack(tmp_path):
    service, _ = make_service(tmp_path, check_interval=0)
    assert service.check()['shouldPrompt']
    assert service.check()['shouldPrompt'] is False
    restarted, _ = make_service(tmp_path)
    assert restarted.check()['shouldPrompt']


def test_no_downgrade_and_missing_highest_package_not_replaced_by_old(tmp_path):
    service, _ = make_service(tmp_path, server=manifest('2.9.0'))
    assert service.check(manual=True)['status'] == 'up-to-date'
    service, _ = make_service(tmp_path/'other', server=manifest('3.1.0', kind='installer'), github=[github_release('3.0.0-preview.2')])
    result=service.check(manual=True)
    assert result['status'] == 'incomplete' and result['packageUnavailable']
    assert result['highestVersion'] == '3.1.0' and result['candidate'] is None


def test_github_list_pagination_not_latest(tmp_path):
    service, transport = make_service(tmp_path, github=[github_release('3.0.0-preview.2')]*100)
    transport.responses[f'{GITHUB_RELEASES}?per_page=100&page=2'] = [github_release('3.0.0-preview.15')]
    result=service.check(manual=True)
    assert result['candidate']['version']=='3.0.0-preview.15'
    assert all('/latest' not in url for url,purpose,_ in transport.calls if purpose=='github')


def test_github_manifest_supports_explicit_platform_and_tag_match(tmp_path):
    release=github_release()
    url='https://github.com/Nephelium/PhoneticToolbox/releases/download/v3.0.0-preview.2/ptb-release.json'
    release['assets']=[{'name':'ptb-release.json','browser_download_url':url}]
    service,transport=make_service(tmp_path,github=[release])
    transport.responses[url]=manifest(source='github')
    assert service.check(manual=True,source='github')['candidate']['source']=='github'
    transport.responses[url]=manifest('3.0.0-preview.3',source='github')
    assert service.check(manual=True)['sources']['github']['code']=='MANIFEST_INVALID'


def test_verified_download_and_native_apply_only_after_recheck(tmp_path):
    applied=[]
    service, _ = make_service(tmp_path, apply_handler=lambda path,kind:applied.append((path,kind)) or {'started':True})
    release=service.check(manual=True)['candidate']['id']
    with pytest.raises(UpdateError,match='确认'):
        service.download(release)
    progress=[]
    result=service.download(release,confirmed=True,progress=progress.append)
    assert result['verified'] and result['sha256']==SHA
    assert 'path' not in result and 'url' not in result
    assert progress[-1]['phase']=='verified'
    path=service.resolve_download(result['downloadId'])
    assert path.read_bytes()==BODY
    with pytest.raises(UpdateError):service.apply(result['downloadId'])
    assert service.apply(result['downloadId'],confirmed=True)=={'started':True}
    assert applied==[(path,'portable')]
    path.write_bytes(b'tampered')
    with pytest.raises(UpdateError):service.apply(result['downloadId'],confirmed=True)
    assert len(applied)==1


@pytest.mark.parametrize('body,headers',[(b'truncated',{}),(BODY+b'extra',{}),(b'x'*len(BODY),{}),(BODY,{'Content-Length':'999'}),(BODY,{'Content-Length':'broken'})])
def test_download_failure_never_produces_executable_or_verification(tmp_path,body,headers):
    service,transport=make_service(tmp_path)
    release=service.check(manual=True)['candidate']['id']
    transport.responses[SERVER_PACKAGE]=(body,headers)
    with pytest.raises(UpdateError):service.download(release,confirmed=True)
    assert not list(tmp_path.rglob('*.zip')) and not list(tmp_path.rglob('verified.json'))
    assert not service._downloads


def test_cancelled_download_is_not_handed_off(tmp_path):
    service,_=make_service(tmp_path)
    release=service.check(manual=True)['candidate']['id']
    cancel=threading.Event()
    def progress(value):cancel.set()
    with pytest.raises(UpdateError,match='取消'):service.download(release,confirmed=True,cancel=cancel,progress=progress)
    assert not service._downloads and not list(tmp_path.rglob('*.zip'))


def test_exact_digest_cross_source_fallback(tmp_path):
    service,transport=make_service(tmp_path,github=[github_release()])
    release=service.check(manual=True)['candidate']['id']
    transport.responses[SERVER_PACKAGE]=TimeoutError('network')
    result=service.download(release,confirmed=True)
    assert result['source']=='github' and result['verified']
    assert any(url==GITHUB_PACKAGE for url,_,_ in transport.calls)


def test_different_build_hash_not_used_as_download_fallback(tmp_path):
    alternate=github_release()
    alternate['assets'][0]['digest']='sha256:'+'0'*64
    service,transport=make_service(tmp_path,github=[alternate])
    result=service.check(manual=True)
    assert result['candidate'] is None and result['status']=='incomplete'
    assert any(s.get('code')=='PACKAGE_CONFLICT' for s in result['sources'].values())
    assert not any(url==GITHUB_PACKAGE for url,_,_ in transport.calls)


def test_download_type_cannot_be_switched_by_browser(tmp_path):
    service,_=make_service(tmp_path)
    release=service.check(manual=True)['candidate']['id']
    with pytest.raises(UpdateError):service.download(release,package_kind='installer',confirmed=True)


def test_preferences_strict_and_release_id_never_url(tmp_path):
    service,transport=make_service(tmp_path)
    for args in ({'source':'evil'},{'channel':'nightly'},{'autoCheck':1},{'url':'https://example.test'}):
        with pytest.raises(UpdateError):service.set_preferences(args)
    with pytest.raises(UpdateError):service.download('https://evil.test',confirmed=True)
    assert not transport.calls
