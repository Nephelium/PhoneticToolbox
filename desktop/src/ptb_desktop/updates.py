"""Owned HTTPS updater. Browser code receives identifiers, never executable URLs/paths.

Release manifest: ptb-release/1 with version, channel, notes, publishedAt and packages
[{kind,platform,name,url,size,sha256}]. Settings/cache live outside the application.
No install/extraction is performed here; the packager supplies the apply callback.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import total_ordering
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, unquote
from urllib.request import Request, build_opener, HTTPRedirectHandler
import uuid

from .platform_paths import user_data_root

SERVER_MANIFEST = 'https://www.phonetictoolbox.com/releases/windows-x64/preview/latest.json'
GITHUB_RELEASES = 'https://api.github.com/repos/Nephelium/PhoneticToolbox/releases'
REGION_URL = 'https://www.cloudflare.com/cdn-cgi/trace'
CHECK_INTERVAL = 6 * 3600
PROMPT_INTERVAL = 24 * 3600
MAX_PACKAGE_SIZE = 8 * 1024 ** 3
_SEMVER = re.compile(r'^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$')


class UpdateError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def normalize_version(value: str) -> str:
    """Only the application's historic a/b/rc notation is normalized."""
    if not isinstance(value, str) or len(value) > 128:
        raise UpdateError('VERSION_INVALID', '版本号格式无效。')
    value = value.strip()
    if value.startswith('v'):
        value = value[1:]
    match = re.fullmatch(r'(\d+\.\d+\.\d+)(a|b|rc)([1-9]\d*|0)', value)
    if match:
        value = f'{match[1]}-{dict(a="alpha", b="beta", rc="rc")[match[2]]}.{match[3]}'
    return value


@total_ordering
class SemVer:
    def __init__(self, value: str):
        self.value = normalize_version(value)
        match = _SEMVER.fullmatch(self.value)
        if not match:
            raise UpdateError('VERSION_INVALID', '版本号不符合 SemVer。')
        self.core = tuple(int(match[i]) for i in (1, 2, 3))
        self.pre = tuple(match[4].split('.')) if match[4] else ()
        if any(p.isdigit() and len(p) > 1 and p[0] == '0' for p in self.pre):
            raise UpdateError('VERSION_INVALID', '预发布版本的数字不能包含前导零。')

    def __eq__(self, other):
        if not isinstance(other, SemVer):
            return NotImplemented
        return self.core == other.core and self.pre == other.pre

    def __lt__(self, other):
        if not isinstance(other, SemVer):
            return NotImplemented
        if self.core != other.core:
            return self.core < other.core
        if not self.pre or not other.pre:
            return bool(self.pre) and not other.pre
        for left, right in zip(self.pre, other.pre):
            if left == right:
                continue
            if left.isdigit() and right.isdigit():
                return int(left) < int(right)
            if left.isdigit() != right.isdigit():
                return left.isdigit()
            return left < right
        return len(self.pre) < len(other.pre)


def _safe_url(url: str, purpose: str) -> str:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except (TypeError, ValueError):
        raise UpdateError('URL_INVALID', '更新地址无效。') from None
    if (parsed.scheme != 'https' or parsed.username or parsed.password or port not in (None, 443)
            or parsed.fragment or any(ord(c) < 32 for c in url) or '\\' in url):
        raise UpdateError('URL_INVALID', '更新仅允许受信任的 HTTPS 地址。')
    path = unquote(parsed.path)
    if any(p in ('.', '..') for p in path.split('/')) or '\\' in path:
        raise UpdateError('URL_INVALID', '更新地址路径无效。')
    host = (parsed.hostname or '').lower()
    if purpose == 'region':
        allowed = host == 'www.cloudflare.com' and path == '/cdn-cgi/trace'
    elif purpose == 'server':
        allowed = host in ('www.phonetictoolbox.com', 'phonetictoolbox.com') and path.startswith('/releases/')
    else:
        allowed = ((host == 'api.github.com' and path.startswith('/repos/Nephelium/PhoneticToolbox/releases'))
                   or (host == 'github.com' and path.startswith('/Nephelium/PhoneticToolbox/releases/'))
                   or (host in ('release-assets.githubusercontent.com', 'objects.githubusercontent.com') and purpose == 'download'))
    if not allowed:
        raise UpdateError('URL_INVALID', '更新地址不在此软件的发布来源内。')
    return url


class _Redirects(HTTPRedirectHandler):
    def __init__(self, purpose):
        self.purpose = purpose

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _safe_url(newurl, self.purpose)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HttpsTransport:
    """Every redirect is checked before any request to its destination."""
    def open(self, url: str, *, purpose: str, timeout: float):
        _safe_url(url, purpose)
        request = Request(url, headers={'User-Agent': 'PhoneticToolbox-Updater/1',
                                       'Accept': 'application/json' if purpose != 'download' else 'application/octet-stream',
                                       'Accept-Encoding': 'identity'})
        if 'api.github.com' == urlsplit(url).hostname:
            request.add_header('X-GitHub-Api-Version', '2022-11-28')
        return build_opener(_Redirects(purpose)).open(request, timeout=timeout)


def _cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise UpdateError('CANCELLED', '更新操作已取消。')


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
    with temp.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _system_country():
    if os.name != 'nt':
        return None
    try:
        buffer = ctypes.create_unicode_buffer(8)
        # LOCALE_SISO3166CTRYNAME, user's Windows region, never an IP heuristic.
        if ctypes.windll.kernel32.GetLocaleInfoEx(None, 0x5A, buffer, len(buffer)):
            country = buffer.value.upper()
            if re.fullmatch('[A-Z]{2}', country):
                return country
    except (AttributeError, OSError):
        pass
    return None


@dataclass(frozen=True)
class Package:
    kind: str
    name: str
    url: str
    size: int
    sha256: str
    source: str

    def public(self):
        return {k: getattr(self, k) for k in ('kind', 'name', 'size', 'sha256', 'source')}


@dataclass
class Release:
    version: SemVer
    source: str
    notes: str
    published_at: str
    packages: list[Package]


class UpdateService:
    def __init__(self, *, current_version='3.0.0-preview.1', data_root=None, transport=None,
                 clock=time.time, monotonic=time.monotonic, system_country=_system_country,
                 check_interval=CHECK_INTERVAL, package_kind='portable', apply_handler=None):
        self.current = SemVer(current_version)
        self.root = Path(data_root) if data_root is not None else user_data_root() / 'updates'
        self.transport = transport or HttpsTransport()
        self.clock, self.monotonic, self.system_country = clock, monotonic, system_country
        self.check_interval = check_interval
        if package_kind not in ('portable', 'installer'):
            raise ValueError('package_kind')
        self.package_kind, self.apply_handler = package_kind, apply_handler
        self._lock = threading.RLock()
        self._check_lock = threading.Lock()
        self._download_lock = threading.Lock()
        self._releases: dict[str, tuple[Release, list[Package]]] = {}
        self._downloads: dict[str, tuple[Path, Package]] = {}
        self._last_result = None
        self._startup_checked = set()
        self._download_versions = {}
        self._maintenance_stop = threading.Event()
        self._maintenance_thread = None
        try:
            state = json.loads((self.root / 'state.json').read_text(encoding='utf-8'))
            self.state = state if isinstance(state, dict) else {}
        except (OSError, ValueError):
            self.state = {}

    def preferences(self):
        with self._lock:
            return {'source': self.state.get('source', 'auto') if self.state.get('source', 'auto') in ('auto', 'server', 'github') else 'auto',
                    'channel': self.state.get('channel', 'preview') if self.state.get('channel', 'preview') in ('preview', 'stable') else 'preview',
                    'autoCheck': self.state.get('autoCheck', True) is not False,
                    'currentVersion': self.current.value, 'packageKind': self.package_kind,
                    'checkIntervalHours': self.check_interval / 3600, 'applyAvailable': self.apply_handler is not None}

    def start_maintenance(self):
        if self._maintenance_thread is not None:
            return
        def maintain():
            from .update_cache import clean_update_cache
            while not self._maintenance_stop.is_set():
                if self._download_lock.acquire(blocking=False):
                    try:
                        clean_update_cache(self.root, protected_downloads=tuple(self._downloads))
                    except Exception:
                        # Maintenance failure never weakens update or close gates.
                        try:
                            from .update_apply import plain_path
                            plain_path(self.root,self.root)
                            _atomic_json(self.root/'cache-status.json',{'state':'failed','checkedAt':time.time(),'code':'CACHE_MAINTENANCE_FAILED','retentionDays':7})
                        except Exception:
                            pass
                    finally:
                        self._download_lock.release()
                if self._maintenance_stop.wait(3600):
                    break
        self._maintenance_thread = threading.Thread(target=maintain, name='ptb-update-cache', daemon=True)
        self._maintenance_thread.start()

    def stop_maintenance(self):
        self._maintenance_stop.set()

    def set_preferences(self, value):
        if not isinstance(value, dict) or set(value) - {'source', 'channel', 'autoCheck'}:
            raise UpdateError('REQUEST_INVALID', '更新设置无效。')
        if ('source' in value and value['source'] not in ('auto', 'server', 'github')
                or 'channel' in value and value['channel'] not in ('preview', 'stable')
                or 'autoCheck' in value and not isinstance(value['autoCheck'], bool)):
            raise UpdateError('REQUEST_INVALID', '更新设置无效。')
        with self._lock:
            self.state.update(value)
            _atomic_json(self.root / 'state.json', self.state)
        return self.preferences()

    def _read(self, url, *, purpose, limit=2 * 1024 ** 2, timeout=8, cancel=None):
        _safe_url(url, purpose)
        deadline = self.monotonic() + timeout
        with self.transport.open(url, purpose=purpose, timeout=min(timeout, 8)) as response:
            data = bytearray()
            while True:
                _cancelled(cancel)
                if self.monotonic() > deadline:
                    raise UpdateError('TIMEOUT', '检查更新超时。')
                chunk = response.read(min(65536, limit + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > limit:
                    raise UpdateError('RESPONSE_TOO_LARGE', '更新信息超过安全大小限制。')
            return bytes(data)

    def _json(self, url, *, purpose, cancel=None):
        try:
            return json.loads(self._read(url, purpose=purpose, cancel=cancel).decode('utf-8'))
        except (ValueError, UnicodeError):
            raise UpdateError('MANIFEST_INVALID', '发布信息无法解析。') from None

    def region(self, cancel=None):
        try:
            # No body, IP, network identity or trace headers enter state/log/errors.
            text = self._read(REGION_URL, purpose='region', limit=8192, timeout=3, cancel=cancel).decode('ascii', 'ignore')
            match = re.search(r'^loc=([A-Z]{2})\s*$', text, flags=re.M)
            if match:
                return {'country': match[1], 'source': 'network', 'label': '网络地区',
                        'preferredSource': 'server' if match[1] == 'CN' else 'github'}
        except (OSError, UpdateError, URLError):
            _cancelled(cancel)
        country = self.system_country()
        return {'country': country, 'source': 'system' if country else 'unknown',
                'label': '系统地区线索' if country else '地区判断不可用',
                'preferredSource': 'server' if country in (None, 'CN') else 'github'}

    def _package(self, item, source):
        if not isinstance(item, dict) or item.get('kind') not in ('portable', 'installer'):
            raise UpdateError('MANIFEST_INVALID', '发布包类型无效。')
        name, size, digest = item.get('name'), item.get('size'), item.get('sha256')
        if (not isinstance(name, str) or len(name) > 160 or not re.fullmatch(r'[\w .()+-]+\.(?:zip|exe)', name, re.I)
                or any(c in name for c in ('/', '\\', ':')) or name.startswith('.')
                or type(size) is not int or not 0 < size <= MAX_PACKAGE_SIZE
                or not isinstance(digest, str) or not re.fullmatch('[0-9a-fA-F]{64}', digest)):
            raise UpdateError('MANIFEST_INVALID', '发布包名称、大小或校验值无效。')
        if item['kind'] == 'installer' and not name.lower().endswith('.exe'):
            raise UpdateError('MANIFEST_INVALID', '安装包格式无效。')
        if item['kind'] == 'portable' and not name.lower().endswith('.zip'):
            raise UpdateError('MANIFEST_INVALID', '免安装包格式无效。')
        url = _safe_url(item.get('url', ''), 'server' if source == 'server' else 'download')
        return Package(item['kind'], name, url, size, digest.lower(), source)

    def _manifest(self, obj, source, channel):
        if not isinstance(obj, dict) or obj.get('schemaVersion') != 'ptb-release/1':
            raise UpdateError('MANIFEST_INVALID', '发布清单版本无法识别。')
        version = SemVer(obj.get('version'))
        if channel == 'stable' and version.pre:
            return None
        if obj.get('channel', 'preview' if version.pre else 'stable') not in ('preview', 'stable'):
            raise UpdateError('MANIFEST_INVALID', '发布通道无效。')
        if obj.get('channel') == 'stable' and version.pre:
            raise UpdateError('MANIFEST_INVALID', '稳定通道不能含预发布版本。')
        packages = obj.get('packages')
        if not isinstance(packages, list) or len(packages) > 16:
            raise UpdateError('MANIFEST_INVALID', '发布包清单无效。')
        parsed = [self._package(item, source) for item in packages if isinstance(item, dict) and item.get('platform') == 'windows-x64']
        if len({p.kind for p in parsed}) != len(parsed):
            raise UpdateError('MANIFEST_INVALID', '发布清单含重复包类型。')
        if not parsed:
            raise UpdateError('MANIFEST_INVALID', '发布清单没有可校验的 Windows 包。')
        notes, published = obj.get('notes', ''), obj.get('publishedAt', '')
        if not isinstance(notes, str) or len(notes) > 50000 or not isinstance(published, str) or len(published) > 80:
            raise UpdateError('MANIFEST_INVALID', '版本说明无效。')
        return Release(version, source, notes, published, parsed)

    def _server(self, channel, cancel):
        url = SERVER_MANIFEST.replace('/preview/', f'/{channel}/')
        obj = self._json(url, purpose='server', cancel=cancel)
        objects = obj.get('releases') if isinstance(obj, dict) and obj.get('schemaVersion') == 'ptb-release-index/1' else [obj]
        if not isinstance(objects, list) or len(objects) > 1000:
            raise UpdateError('MANIFEST_INVALID', '发布索引无效。')
        return [release for item in objects if (release := self._manifest(item, 'server', channel))]

    def _github(self, channel, cancel):
        releases = []
        deadline = self.monotonic() + 30
        # A bounded full list, including prereleases. Truncation is an error, not "latest".
        for page in range(1, 11):
            _cancelled(cancel)
            if self.monotonic() > deadline:
                raise UpdateError('TIMEOUT', 'GitHub 发布列表检查超时。')
            items = self._json(f'{GITHUB_RELEASES}?per_page=100&page={page}', purpose='github', cancel=cancel)
            if not isinstance(items, list):
                raise UpdateError('MANIFEST_INVALID', 'GitHub 发布列表无效。')
            for item in items:
                if not isinstance(item, dict) or item.get('draft'):
                    continue
                try:
                    version = SemVer(item.get('tag_name', ''))
                except UpdateError:
                    continue
                if channel == 'stable' and (version.pre or item.get('prerelease')):
                    continue
                assets = item.get('assets', [])
                if not isinstance(assets, list):
                    raise UpdateError('MANIFEST_INVALID', 'GitHub 发布资源无效。')
                manifest = next((a for a in assets if isinstance(a, dict) and a.get('name') == 'ptb-release.json'), None)
                if manifest:
                    manifest_url = _safe_url(manifest.get('browser_download_url', ''), 'download')
                    release = self._manifest(self._json(manifest_url, purpose='download', cancel=cancel), 'github', channel)
                    if release and release.version != version:
                        raise UpdateError('MANIFEST_INVALID', 'GitHub 标签与清单版本不一致。')
                    if release:
                        releases.append(release)
                    continue
                parsed = []
                for asset in assets:
                    if not isinstance(asset, dict):
                        continue
                    name, digest = asset.get('name', ''), asset.get('digest', '')
                    if not isinstance(name, str) or not isinstance(digest, str) or not digest.startswith('sha256:'):
                        continue
                    lowered = name.lower()
                    # Digest-only releases must identify the supported architecture.
                    if not ('windows-x64' in lowered or 'win-x64' in lowered or 'win64' in lowered):
                        continue
                    kind = 'portable' if 'portable' in lowered and lowered.endswith('.zip') else 'installer' if ('setup' in lowered or 'installer' in lowered) and lowered.endswith('.exe') else None
                    if kind:
                        parsed.append(self._package({'kind': kind, 'name': name, 'size': asset.get('size'),
                                                     'sha256': digest[7:], 'url': asset.get('browser_download_url')}, 'github'))
                if parsed:
                    releases.append(Release(version, 'github', str(item.get('body') or '')[:50000], str(item.get('published_at') or '')[:80], parsed))
                else:
                    # A recognizable release without integrity information is unknown.
                    raise UpdateError('CHECKSUM_MISSING', 'GitHub 发布缺少可验证的 Windows 更新包。')
            if len(items) < 100:
                return releases
        raise UpdateError('LIST_INCOMPLETE', 'GitHub 发布列表超过检查范围，无法确认最高版本。')

    @staticmethod
    def _failure(error):
        if isinstance(error, UpdateError):
            return {'status': 'error', 'code': error.code, 'message': str(error)}
        if isinstance(error, HTTPError):
            return {'status': 'error', 'code': 'HTTP_ERROR', 'message': f'发布来源返回 HTTP {error.code}。'}
        if isinstance(error, (TimeoutError, OSError, URLError)):
            return {'status': 'error', 'code': 'NETWORK_ERROR', 'message': '连接失败或超时，请稍后重试。'}
        return {'status': 'error', 'code': 'CHECK_FAILED', 'message': '更新检查失败，请稍后重试。'}

    def check(self, *, manual=False, source=None, channel=None, cancel=None):
        if not isinstance(manual, bool) or source not in (None, 'auto', 'server', 'github') or channel not in (None, 'stable', 'preview'):
            raise UpdateError('REQUEST_INVALID', '检查更新参数无效。')
        with self._check_lock:
            prefs = self.preferences()
            source, channel = source or prefs['source'], channel or prefs['channel']
            now = self.clock()
            key = f'{self.current.value}|{channel}|{source}'
            with self._lock:
                # Every new process checks both sources. Repeated component mounts
                # in this process reuse the first check without another network trip.
                recent = key in self._startup_checked
            if not manual and (not prefs['autoCheck'] or recent):
                result = dict(self._last_result or {'currentVersion': self.current.value, 'sources': {}, 'candidate': None})
                result.update(status='deferred', reason='disabled' if not prefs['autoCheck'] else 'interval', shouldPrompt=False)
                return result
            _cancelled(cancel)
            region = self.region(cancel) if source == 'auto' else {'source': 'manual', 'country': None, 'label': '手动选择', 'preferredSource': source}
            preferred = region['preferredSource']
            results, all_releases = {}, []
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix='ptb-update-check') as pool:
                jobs = {name: pool.submit(fn, channel, cancel) for name, fn in (('server', self._server), ('github', self._github))}
                for name, job in jobs.items():
                    try:
                        releases = job.result()
                        all_releases.extend(releases)
                        results[name] = {'status': 'checked' if releases else 'no-releases', 'count': len(releases),
                                         'version': max((r.version for r in releases), default=None).value if releases else None}
                    except Exception as error:
                        _cancelled(cancel)
                        results[name] = self._failure(error)
            _cancelled(cancel)
            best = max(all_releases, key=lambda r: (r.version, any(p.kind == self.package_kind for p in r.packages), r.source == preferred), default=None)
            candidate = None
            unavailable = bool(best and best.version > self.current and not any(p.kind == self.package_kind for p in best.packages))
            conflict = False
            if best:
                selected = {(p.kind, p.size, p.sha256) for p in best.packages if p.kind == self.package_kind}
                for alternate in all_releases:
                    other = {(p.kind, p.size, p.sha256) for p in alternate.packages if p.kind == self.package_kind}
                    if alternate.version == best.version and selected and other and selected != other:
                        conflict = True
                        results[alternate.source] = {'status': 'error', 'code': 'PACKAGE_CONFLICT', 'message': '同版本两来源的更新包摘要不一致，已阻止换版。'}
            if best and best.version > self.current and not unavailable and not conflict:
                release_id = uuid.uuid4().hex
                packages = list(best.packages)
                for alternate in all_releases:
                    if alternate is not best and alternate.version == best.version:
                        packages.extend(alternate.packages)
                self._releases[release_id] = best, packages
                candidate = {'id': release_id, 'version': best.version.value, 'source': best.source,
                             'notes': best.notes, 'publishedAt': best.published_at,
                             'packages': [p.public() for p in best.packages]}
            failed = any(r['status'] == 'error' for r in results.values())
            status = 'available' if candidate else 'incomplete' if failed or unavailable or best is None else 'up-to-date'
            with self._lock:
                prompts = self.state.get('prompts', {})
                prompted = prompts.get(candidate['version']) if candidate and isinstance(prompts, dict) else None
                should_prompt = bool(candidate and (manual or type(prompted) not in (int, float) or now - prompted < 0 or now - prompted >= PROMPT_INTERVAL))
                self.state['lastCheck'] = {'key': key, 'time': now}
                self._startup_checked.add(key)
                _atomic_json(self.root / 'state.json', self.state)
            result = {'status': status, 'currentVersion': self.current.value, 'channel': channel,
                      'checkedAt': now, 'region': region, 'preferredSource': preferred,
                      'sources': results, 'candidate': candidate, 'shouldPrompt': should_prompt,
                      'packageKind': self.package_kind, 'packageUnavailable': unavailable,
                      'highestVersion': best.version.value if best else None}
            self._last_result = result
            return result

    def acknowledge_notice(self, release_id):
        if not isinstance(release_id, str):
            raise UpdateError('RELEASE_UNKNOWN', '版本信息已失效，请重新检查。')
        release = self._releases.get(release_id)
        if not release:
            raise UpdateError('RELEASE_UNKNOWN', '版本信息已失效，请重新检查。')
        with self._lock:
            if not isinstance(self.state.get('prompts'), dict):
                self.state['prompts'] = {}
            self.state['prompts'][release[0].version.value] = self.clock()
            _atomic_json(self.root / 'state.json', self.state)
        return {'acknowledged': True}

    def download(self, release_id, *, package_kind=None, confirmed=False, cancel=None, progress=None):
        if confirmed is not True:
            raise UpdateError('CONFIRM_REQUIRED', '请确认后再下载更新。')
        package_kind = package_kind or self.package_kind
        if package_kind not in ('portable', 'installer') or package_kind != self.package_kind:
            raise UpdateError('PACKAGE_KIND', '更新包类型与当前安装方式不一致。')
        entry = self._releases.get(release_id) if isinstance(release_id, str) else None
        if not entry:
            raise UpdateError('RELEASE_UNKNOWN', '版本信息已失效，请重新检查。')
        packages = [p for p in entry[1] if p.kind == package_kind]
        if not packages:
            raise UpdateError('PACKAGE_UNAVAILABLE', '此版本没有匹配的更新包。')
        primary = packages[0]
        # Fallback can only fetch the exact expected bytes of the selected build.
        packages = [p for p in packages if p.sha256 == primary.sha256 and p.size == primary.size]
        if not self._download_lock.acquire(blocking=False):
            raise UpdateError('DOWNLOAD_BUSY', '已有更新包正在下载。')
        try:
            errors = []
            for package in packages:
                try:
                    result = self._download_package(package, cancel, progress)
                    self._download_versions[result['downloadId']] = entry[0].version.value
                    return result
                except Exception as error:
                    _cancelled(cancel)
                    failure = self._failure(error)
                    errors.append({'source': package.source, **failure})
                    if progress:
                        progress({'phase': 'fallback', 'source': package.source, 'received': 0, 'total': package.size, 'message': failure['message']})
            raise UpdateError('DOWNLOAD_FAILED', '下载或校验失败。' + ' '.join(f'{e["source"]}: {e["message"]}' for e in errors))
        finally:
            self._download_lock.release()

    def _download_package(self, package, cancel, progress):
        download_id = uuid.uuid4().hex
        if (self.root / 'downloads').is_symlink():
            raise UpdateError('CACHE_INVALID', '更新缓存目录无效。')
        folder = self.root / 'downloads' / download_id
        folder.mkdir(parents=True)
        temp, target = folder / 'payload.partial', folder / package.name
        digest, received = hashlib.sha256(), 0
        deadline = self.monotonic() + 30 * 60
        try:
            with self.transport.open(package.url, purpose='server' if package.source == 'server' else 'download', timeout=8) as response, temp.open('xb') as stream:
                content_length = response.headers.get('Content-Length')
                if content_length is not None and (not str(content_length).isdigit() or int(content_length) != package.size):
                    raise UpdateError('SIZE_MISMATCH', '下载响应大小与发布清单不一致。')
                while True:
                    _cancelled(cancel)
                    if self.monotonic() > deadline:
                        raise UpdateError('TIMEOUT', '更新包下载超时。')
                    chunk = response.read(min(1024 * 256, package.size + 1 - received))
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > package.size:
                        raise UpdateError('SIZE_MISMATCH', '下载内容超过发布清单大小。')
                    digest.update(chunk)
                    stream.write(chunk)
                    if progress:
                        progress({'phase': 'downloading', 'source': package.source, 'received': received, 'total': package.size})
                stream.flush()
                os.fsync(stream.fileno())
            _cancelled(cancel)
            if received != package.size:
                raise UpdateError('SIZE_MISMATCH', '更新包未完整下载。')
            if digest.hexdigest() != package.sha256:
                raise UpdateError('HASH_MISMATCH', '更新包校验不一致，已阻止使用。')
            os.replace(temp, target)
            _atomic_json(folder / 'verified.json', {'name': package.name, 'sha256': package.sha256, 'size': package.size, 'source': package.source})
            self._downloads[download_id] = target, package
            if progress:
                progress({'phase': 'verified', 'source': package.source, 'received': received, 'total': package.size})
            return {'downloadId': download_id, 'name': package.name, 'size': package.size, 'sha256': package.sha256,
                    'source': package.source, 'kind': package.kind, 'verified': True, 'applyAvailable': self.apply_handler is not None}
        except Exception:
            # Failed content remains non-executable in the owned cache for root's cleanup policy.
            if temp.exists():
                os.replace(temp, folder / 'payload.failed')
            raise

    def resolve_download(self, download_id):
        """Native packager boundary: revalidate bytes immediately before handoff."""
        entry = self._downloads.get(download_id) if isinstance(download_id, str) else None
        if not entry:
            raise UpdateError('DOWNLOAD_UNKNOWN', '下载记录已失效，请重新下载。')
        path, package = entry
        if (path.is_symlink() or path.parent.is_symlink() or (self.root / 'downloads').is_symlink()
                or path.resolve().parent != self.root.resolve() / 'downloads' / download_id):
            raise UpdateError('CACHE_INVALID', '更新缓存位置无效。')
        try:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                size = 0
                for chunk in iter(lambda: stream.read(1024 * 256), b''):
                    digest.update(chunk)
                    size += len(chunk)
            if size != package.size or digest.hexdigest() != package.sha256:
                raise UpdateError('CACHE_INVALID', '更新缓存已变化，已阻止使用。')
        except OSError:
            raise UpdateError('CACHE_INVALID', '更新缓存无法读取。') from None
        return path

    def apply(self, download_id, *, confirmed=False):
        if confirmed is not True:
            raise UpdateError('CONFIRM_REQUIRED', '请确认退出并更新后再继续。')
        if self.apply_handler is None:
            raise UpdateError('APPLY_UNAVAILABLE', '当前运行方式尚未接入换版，请保留已下载包。')
        path = self.resolve_download(download_id)
        package = self._downloads[download_id][1]
        return self.apply_handler(path, package.kind)
