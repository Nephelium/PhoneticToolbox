"""Pinned, explicitly trusted optional-component installation, with no deletion.

The caller must obtain trusted metadata out-of-band. A manifest inside an archive
is never a trust anchor. Production currently has no published catalog.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sysconfig
import urllib.request
import urllib.parse
from uuid import uuid4
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name):
    if not isinstance(name, str) or not name or len(name) > 1024 or '\\' in name or ':' in name or '\x00' in name:
        raise ValueError('m11_unsafe_path')
    p = PurePosixPath(name)
    if p.is_absolute() or any(part in ('', '.', '..') for part in name.split('/')):
        raise ValueError('m11_unsafe_path')
    for part in p.parts:
        if part.endswith((' ', '.')) or any(ord(c) < 32 for c in part) or re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', part, re.I):
            raise ValueError('m11_unsafe_path')
    return p


def no_links(path):
    p = Path(path).absolute()
    for item in (p, *p.parents):
        if item.is_symlink() or getattr(item, 'is_junction', lambda: False)():
            raise ValueError('m11_component_link')


def atomic_json(path, value):
    path = Path(path)
    no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.part')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def default_root():
    base = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / '.local/share')))
    return base / 'PhoneticToolbox/v3/components/mfa'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('m11_download_redirect_rejected')


class ComponentManager:
    def __init__(self, root=None):
        self.root = Path(root or default_root()).absolute()
        no_links(self.root)

    def import_archive(self, archive, trusted, self_test):
        if trusted.get('schema') != 'm11-component/1' or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', trusted.get('id', '')):
            raise ValueError('m11_manifest_invalid')
        expected_platform = 'windows' if os.name == 'nt' else 'linux'
        arch = platform.machine().lower().replace('amd64', 'x86_64')
        if not arch and sysconfig.get_platform() == 'win-amd64':
            arch = 'x86_64'
        if trusted.get('platform') != expected_platform or trusted.get('arch') != arch:
            raise ValueError('m11_platform_mismatch')
        if trusted.get('mfa_version') != '3.3.8' or not trusted.get('source') or not isinstance(trusted.get('dependencies'), list):
            raise ValueError('m11_manifest_invalid')
        archive = Path(archive)
        no_links(archive)
        if archive.stat().st_size != trusted.get('download_bytes') or digest(archive) != trusted.get('sha256'):
            raise ValueError('m11_archive_hash_mismatch')
        records = trusted.get('files', [])
        expected = {}
        total = 0
        for row in records:
            name = str(safe_name(row['path']))
            if name.casefold() in expected or type(row['bytes']) is not int or row['bytes'] < 0 or not re.fullmatch(r'[a-f0-9]{64}', row['sha256']):
                raise ValueError('m11_manifest_invalid')
            expected[name.casefold()] = row
            total += row['bytes']
        if not expected or len(expected) > 150000 or total != trusted.get('installed_bytes') or total > 15_000_000_000:
            raise ValueError('m11_component_budget')
        import shutil
        self.root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.root).free < total + 512_000_000 + 64_000_000:
            raise ValueError('m11_component_disk_full')
        # This IS the permanent version path. No post-relocation rename.
        target = self.root / 'versions' / ('mfa338-' + uuid4().hex[:16])
        if expected_platform == 'windows' and any(len(str(target / row['path']))>=240 for row in records if row['path'].lower().endswith(('.pyd','.dll','.exe'))):
            raise ValueError('m11_install_path_too_long')
        no_links(target)
        target.mkdir(parents=True)
        with zipfile.ZipFile(archive) as z:
            members = {}
            for info in z.infolist():
                name = str(safe_name(info.filename.rstrip('/')))
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)) or info.flag_bits & 1:
                    raise ValueError('m11_unsafe_archive')
                if info.is_dir():
                    continue
                key = name.casefold()
                if key in members or key not in expected or expected[key]['path'] != name or expected[key]['bytes'] != info.file_size:
                    raise ValueError('m11_archive_manifest_mismatch')
                members[key] = info
            if set(members) != set(expected):
                raise ValueError('m11_archive_manifest_mismatch')
            for key, info in members.items():
                row = expected[key]
                destination = target / row['path']
                destination.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                count = 0
                with z.open(info) as source, destination.open('xb') as output:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        count += len(block)
                        if count > row['bytes']:
                            raise ValueError('m11_component_budget')
                        output.write(block)
                        h.update(block)
                if count != row['bytes'] or h.hexdigest() != row['sha256']:
                    raise ValueError('m11_file_hash_mismatch')
                if expected_platform == 'linux' and (info.external_attr >> 16) & 0o111:
                    destination.chmod(0o700)
        # Explicit trusted fixed relocation, when declared, belongs in self_test.
        # Unsuccessful candidates remain inspectable and never replace current.
        receipt = self_test(target)
        if not isinstance(receipt, dict) or receipt.get('success') is not True:
            raise ValueError('m11_self_test_failed')
        atomic_json(target / 'ptb-component.json', dict(manifest=trusted, receipt=receipt))
        atomic_json(self.root / 'current.json', dict(id=trusted['id'], path=str(target), receipt=receipt))
        return target

    def download(self, trusted):
        url = trusted.get('url')
        if not url:
            raise ValueError('m11_download_not_published')
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.fragment or parsed.query or parsed.hostname != trusted.get('trusted_host') or 'latest' in parsed.path.lower():
            raise ValueError('m11_download_source_rejected')
        sha, size = trusted.get('sha256', ''), trusted.get('download_bytes')
        if not re.fullmatch(r'[a-f0-9]{64}', sha) or type(size) is not int or not 0 < size <= 10_000_000_000:
            raise ValueError('m11_manifest_invalid')
        cache = self.root / 'downloads'
        no_links(cache)
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (sha + '.part')
        no_links(path)
        offset = path.stat().st_size if path.exists() else 0
        if offset > size:
            raise ValueError('m11_download_size_mismatch')
        if offset < size:
            request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-', 'Accept-Encoding': 'identity'})
            with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
                if offset and (response.status != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{size - 1}/{size}'):
                    raise ValueError('m11_download_resume_rejected')
                with path.open('ab') as stream:
                    while chunk := response.read(min(1024 * 1024, size - offset + 1)):
                        offset += len(chunk)
                        if offset > size:
                            raise ValueError('m11_download_size_mismatch')
                        stream.write(chunk)
        if offset != size or digest(path) != sha:
            raise ValueError('m11_archive_hash_mismatch')
        return path
