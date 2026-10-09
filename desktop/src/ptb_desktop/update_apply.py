"""Frozen-only, owned Windows update handoff. No old-version or user-data deletion."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
import zipfile

from .platform_paths import user_data_root
from .startup_cache import external_executable
from .updates import SemVer, UpdateError, _atomic_json

MAX_FILES = 80000
MAX_EXPANDED = 20 * 1024 ** 3
MAX_MEMBER = 4 * 1024 ** 3
MAX_RATIO = 1000
APP_DIR = 'PhoneticToolbox'
APP_EXE = 'PhoneticToolbox.exe'


def fail(code, message):
    raise UpdateError(code, message)


def plain_path(path, root):
    """Reject symlinks and Windows junctions throughout the owned boundary."""
    path, root = Path(path).absolute(), Path(root).absolute()
    if not path.is_relative_to(root):
        fail('APPLY_PATH', '更新文件位置无效。')
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            attrs = getattr(item.lstat(), 'st_file_attributes', 0)
            if item.is_symlink() or attrs & 0x400:
                fail('APPLY_PATH', '更新目录包含链接，已阻止使用。')
        if item == root:
            break
    if not path.resolve().is_relative_to(root.resolve()):
        fail('APPLY_PATH', '更新文件越出所属目录。')
    return path


def owned_tree(path, root):
    pending = [Path(path)]
    while pending:
        item = plain_path(pending.pop(), root)
        yield item
        if item.is_dir():
            # Check each reparse point before traversal, never enumerate its target.
            for child in item.iterdir():
                plain_path(child, root)
                pending.append(child)


def verify_bytes(path, size, sha256):
    if type(size) is not int or size < 0 or not isinstance(sha256, str) or not re.fullmatch('[0-9a-f]{64}', sha256):
        fail('APPLY_METADATA', '更新校验信息无效。')
    try:
        with Path(path).open('rb') as stream:
            if os.fstat(stream.fileno()).st_size != size:
                fail('APPLY_HASH', '更新包大小已变化。')
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != sha256:
            fail('APPLY_HASH', '更新包摘要已变化。')
    except OSError:
        fail('APPLY_READ', '更新包无法读取，原程序保留。')


def read_json(path, limit=1024 * 1024):
    if Path(path).stat().st_size > limit:
        fail('APPLY_METADATA', '更新说明文件超过限制。')
    value = json.loads(Path(path).read_text('utf-8'))
    if not isinstance(value, dict):
        fail('APPLY_METADATA', '更新说明文件无效。')
    return value


def executable_magic(path):
    with Path(path).open('rb') as stream:
        return stream.read(2) == b'MZ'


def release_layout(app_root, version, *, require_runtime=True):
    app_root = Path(app_root)
    value = read_json(plain_path(app_root / 'application.json', app_root))
    if value.get('schema') != 'ptb-desktop-release/1' or value.get('version') != version or value.get('entry') != APP_EXE:
        fail('APPLY_LAYOUT', '更新应用标识或版本与清单不一致。')
    SemVer(version)
    exe = plain_path(app_root / APP_EXE, app_root)
    if not exe.is_file() or exe.stat().st_size < 2 or not executable_magic(exe):
        fail('APPLY_LAYOUT', '更新包缺少有效的应用程序。')
    layout = value.get('layout', 'onedir/1')
    if layout == 'onefile/1':
        metadata = value.get('executable', {})
        if not isinstance(metadata, dict):
            fail('APPLY_LAYOUT', '单文件更新缺少程序校验信息。')
        verify_bytes(exe, metadata.get('size'), metadata.get('sha256'))
    elif layout != 'onedir/1':
        fail('APPLY_LAYOUT', '更新包运行布局不受支持。')
    elif require_runtime:
        from .bundle_manifest import runtime_bindings
        runtime_bindings(plain_path(app_root / '_internal', app_root), host_platform='win32', host_arch='x86_64')
    return exe


def package_kind(executable=None):
    exe = Path(executable or external_executable())
    marker = exe.parent / '.ptb-installed.json'
    if not marker.exists():
        return 'portable'
    value = read_json(plain_path(marker, exe.parent))
    if value.get('schema') != 'ptb-install/1' or value.get('kind') != 'installer':
        fail('APPLY_LAYOUT', '安装方式标识无效，请重新安装此应用。')
    return 'installer'


def archive_members(archive):
    members, paths, total = [], {}, 0
    for info in archive.infolist():
        name = info.orig_filename
        # Validate the original ZIP spelling before pathlib can normalize it.
        if len(members) >= MAX_FILES or not name or '\\' in name or '\x00' in name or name.startswith('/') or '//' in name:
            fail('ZIP_PATH', '更新包路径无效。')
        parts = name.rstrip('/').split('/')
        if parts[0] != APP_DIR or any(not part or part in ('.', '..') or part.endswith((' ', '.')) or
                any(ord(c) < 32 or c in '<>:"|?*' for c in part) or
                re.fullmatch(r'(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', part, re.I) for part in parts):
            fail('ZIP_PATH', '更新包包含非法 Windows 路径。')
        if len(name) > 1024 or any(len(part) > 255 for part in parts):
            fail('ZIP_PATH', '更新包路径超过限制。')
        key = '/'.join(parts).casefold()
        mode = info.external_attr >> 16
        if key in paths or info.flag_bits & 1 or info.external_attr & 0x400 or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
            fail('ZIP_MEMBER', '更新包存在重复路径、链接或不支持的成员。')
        for i in range(1, len(parts)):
            if paths.get('/'.join(parts[:i]).casefold()) is False:
                fail('ZIP_MEMBER', '更新包文件与目录冲突。')
        if not info.is_dir() and any(k.startswith(key + '/') for k in paths):
            fail('ZIP_MEMBER', '更新包文件与目录冲突。')
        if info.file_size > MAX_MEMBER or info.file_size > max(1, info.compress_size) * MAX_RATIO:
            fail('ZIP_BUDGET', '更新包展开大小超过安全限制。')
        total += info.file_size
        if total > MAX_EXPANDED:
            fail('ZIP_BUDGET', '更新包展开总量超过安全限制。')
        paths[key] = info.is_dir()
        members.append((info, parts))
    required = {'phonetictoolbox/application.json', 'phonetictoolbox/phonetictoolbox.exe'}
    if not required.issubset(paths) or any(paths[p] for p in required):
        fail('ZIP_LAYOUT', '更新包缺少完整的应用布局。')
    metadata_info = next(info for info, parts in members if '/'.join(parts).casefold() == 'phonetictoolbox/application.json')
    if metadata_info.file_size > 1024 * 1024:
        fail('ZIP_LAYOUT', '更新包说明超过限制。')
    value = json.loads(archive.read(metadata_info))
    layout = value.get('layout', 'onedir/1') if isinstance(value, dict) else None
    if layout == 'onedir/1' and paths.get('phonetictoolbox/_internal/desktop-bundle.json') is not False:
        fail('ZIP_LAYOUT', '更新包缺少完整的运行文件。')
    if layout not in ('onedir/1', 'onefile/1'):
        fail('ZIP_LAYOUT', '更新包运行布局不受支持。')
    return members, total


def extract_portable(package, destination, version):
    """Destination must be a new sibling. Existing programs are never overwritten."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        fail('APPLY_DESTINATION', '新版本目录已经存在，旧版保留。')
    with zipfile.ZipFile(package) as archive:
        members, total = archive_members(archive)
        if shutil.disk_usage(destination.parent).free < total + 64 * 1024 ** 2:
            fail('APPLY_SPACE', '新版本目录空间不足，旧版保留。')
        destination.mkdir()
        for info, parts in members:
            target = plain_path(destination.joinpath(*parts), destination)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            received = 0
            with archive.open(info) as source, target.open('xb') as output:
                while block := source.read(256 * 1024):
                    received += len(block)
                    if received > info.file_size:
                        fail('ZIP_BUDGET', '更新包成员超出声明大小。')
                    output.write(block)
            if received != info.file_size:
                fail('ZIP_MEMBER', '更新包成员未完整展开。')
    return release_layout(destination / APP_DIR, version)


def kernel():
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    api.OpenProcess.restype = wintypes.HANDLE
    api.GetProcessId.argtypes = (wintypes.HANDLE,)
    api.GetProcessId.restype = wintypes.DWORD
    api.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.CloseHandle.argtypes = (wintypes.HANDLE,)
    return api


def open_parent_handle():
    handle = kernel().OpenProcess(0x00100000 | 0x1000, True, os.getpid())
    if not handle:
        fail('APPLY_PROCESS', '无法保护当前进程身份，已阻止更新。')
    return int(handle)


def wait_parent(handle, pid, timeout_ms=180000):
    api = kernel()
    if api.GetProcessId(handle) != pid:
        fail('APPLY_PROCESS', '原程序进程身份不匹配。')
    if api.WaitForSingleObject(handle, timeout_ms) != 0:
        fail('APPLY_PROCESS', '原程序尚未退出，未运行更新。')


def stage_helper(root, origin):
    """Relocate only the trusted freeze bootstrap to avoid installed DLL locks."""
    folder = plain_path(Path(root) / 'helpers' / uuid.uuid4().hex, root)
    folder.mkdir(parents=True)
    origin = Path(origin).resolve()
    helper = folder / APP_EXE
    shutil.copy2(origin, helper)
    verify_bytes(helper, origin.stat().st_size, hashlib.sha256(origin.read_bytes()).hexdigest())
    internal = origin.parent / '_internal'
    if not internal.is_dir():
        # A direct one-file EXE carries its own Python/Qt bootstrap. A relocated
        # portable EXE may have no neighbouring application.json at all.
        own_frozen = getattr(sys, 'frozen', False) and origin == external_executable().resolve()
        embedded = Path(getattr(sys, '_MEIPASS', origin.parent)) / 'desktop-bundle.json'
        if own_frozen and embedded.is_file():
            descriptor = read_json(embedded)
            onefile = descriptor.get('runtimeArchive', {}).get('schema') == 'ptb-runtime-archive/1'
        elif (origin.parent / 'application.json').is_file():
            descriptor = read_json(origin.parent / 'application.json')
            onefile = descriptor.get('layout') == 'onefile/1'
            if onefile:
                release_layout(origin.parent, descriptor.get('version'))
        else:
            onefile = False
        if not onefile:
            fail('APPLY_LAYOUT', '当前冻结程序缺少运行文件。')
    target = folder / '_internal'
    if internal.is_dir():
        target.mkdir()
    for item in internal.iterdir() if internal.is_dir() else ():
        if item.is_file():
            plain_path(item, internal)
            shutil.copy2(item, target / item.name)
    # PyInstaller's startup hooks run before our fixed helper entry. pkg_resources
    # reads setuptools' data, and the Qt hook registers an embedded qt.conf via
    # QtCore. Preserve that small bootstrap without scientific runtimes, WebEngine
    # resources or a GUI/application instance.
    resources = internal / 'setuptools'
    if resources.is_dir():
        for item in owned_tree(resources, internal):
            pass
        shutil.copytree(resources, target / 'setuptools', symlinks=False)
    qt = internal / 'PyQt6'
    if qt.is_dir():
        qt_target = target / 'PyQt6'
        (qt_target / 'Qt6' / 'plugins').mkdir(parents=True)
        for item in qt.iterdir():
            if item.is_file():
                plain_path(item, internal)
                shutil.copy2(item, qt_target / item.name)
    # The trusted fixed entry imports this source snapshot.
    source = internal / 'desktop' / 'src'
    if source.is_dir():
        for item in owned_tree(source, internal):
            pass
        shutil.copytree(source, target / 'desktop' / 'src', symlinks=False)
    files = []
    for item in owned_tree(folder, folder):
        if item.is_file():
            files.append({'path': item.relative_to(folder).as_posix(), 'size': item.stat().st_size,
                          'sha256': hashlib.sha256(item.read_bytes()).hexdigest()})
    _atomic_json(folder / 'helper-files.json', {'schema': 'ptb-update-helper/1', 'files': files})
    for item in files:
        verify_bytes(folder / item['path'], item['size'], item['sha256'])
    if sys.platform == 'win32' and getattr(sys, 'frozen', False):
        # With no request arguments, the fixed helper entry returns 2 after all
        # frozen startup hooks have succeeded. Check before closing the GUI.
        probe = subprocess.Popen([str(helper), '--ptb-apply-update'],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            code = probe.wait(timeout=30)
        except subprocess.TimeoutExpired:
            probe.kill()
            probe.wait(timeout=10)
            fail('APPLY_BOOTSTRAP', '更新辅助程序未能完成启动检查，当前程序保持打开。')
        if code != 2:
            fail('APPLY_BOOTSTRAP', '更新辅助程序启动检查失败，当前程序保持打开。')
    return helper


def prepare_handoff(root, path, kind, size, sha256, version, current_version, origin=None):
    root = Path(root).absolute()
    plain_path(root, root)
    origin = Path(origin or external_executable()).resolve()
    path = plain_path(path, root / 'downloads')
    verify_bytes(path, size, sha256)
    if SemVer(version) <= SemVer(current_version) or kind not in ('portable', 'installer') or package_kind(origin) != kind:
        fail('APPLY_METADATA', '更新版本或安装方式不匹配。')
    if kind == 'portable':
        with zipfile.ZipFile(path) as archive:
            archive_members(archive)
            info = archive.getinfo(APP_DIR + '/application.json')
            if info.file_size > 1024 * 1024:
                fail('APPLY_METADATA', '更新说明文件超过限制。')
            release = json.loads(archive.read(info))
            if release.get('schema') != 'ptb-desktop-release/1' or release.get('version') != version or release.get('entry') != APP_EXE:
                fail('APPLY_LAYOUT', '更新包应用版本与发布清单不一致。')
    elif not executable_magic(path):
        fail('APPLY_LAYOUT', '安装程序格式无效。')
    helper = stage_helper(root, origin)
    folder = plain_path(root / 'apply' / uuid.uuid4().hex, root)
    folder.mkdir(parents=True)
    request = folder / 'request.json'
    plan = {'schema': 'ptb-apply-update/1', 'token': uuid.uuid4().hex, 'kind': kind, 'size': size,
            'sha256': sha256, 'version': version, 'currentVersion': current_version,
            'package': str(path), 'originExe': str(origin), 'helperExe': str(helper),
            'helperSha256': hashlib.sha256(helper.read_bytes()).hexdigest(), 'helperSize': helper.stat().st_size,
            'helperFilesSha256': hashlib.sha256((helper.parent / 'helper-files.json').read_bytes()).hexdigest(),
            'userData': str(user_data_root()), 'parentPid': os.getpid()}
    _atomic_json(request, plan)
    _atomic_json(folder / 'status.json', {'state': 'awaiting-close', 'version': version})
    return request, plan


def launch_handoff(request, plan):
    verify_bytes(plan['package'], plan['size'], plan['sha256'])
    manifest = Path(plan['helperExe']).parent / 'helper-files.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != plan['helperFilesSha256']:
        fail('APPLY_HASH', 'helper 文件登记已变化。')
    for item in read_json(manifest)['files']:
        verify_bytes(manifest.parent / item['path'], item['size'], item['sha256'])
    handle = open_parent_handle()
    info = subprocess.STARTUPINFO()
    info.lpAttributeList = {'handle_list': [handle]}
    try:
        _atomic_json(Path(request).parent / 'closed.json', {'schema': 'ptb-update-closed/1', 'token': plan['token']})
        return subprocess.Popen([plan['helperExe'], '--ptb-apply-update', str(request), plan['token'], str(handle)],
                                close_fds=True, startupinfo=info, creationflags=subprocess.CREATE_NO_WINDOW)
    finally:
        kernel().CloseHandle(handle)


def run_handoff(request, token, handle, *, root=None, executable=None, wait=wait_parent, launch=subprocess.Popen):
    """Adapters are Python test boundaries; no environment/CLI dry-run bypass."""
    root = Path(root or user_data_root() / 'updates').absolute()
    plain_path(root, root)
    request = plain_path(request, root / 'apply')
    if request.name != 'request.json' or not re.fullmatch('[0-9a-f]{32}', request.parent.name) or request.parent.parent != root / 'apply':
        fail('APPLY_PATH', '换版请求位置无效。')
    status = request.parent / 'status.json'
    try:
        plan = read_json(request)
        if (plan.get('schema') != 'ptb-apply-update/1' or plan.get('token') != token or
                plan.get('userData') != str(user_data_root()) or type(plan.get('parentPid')) is not int):
            fail('APPLY_METADATA', '换版请求身份无效。')
        helper = plain_path(plan['helperExe'], root / 'helpers')
        if helper.resolve() != Path(executable or external_executable()).resolve():
            fail('APPLY_PROCESS', '换版 helper 身份无效。')
        verify_bytes(helper, plan['helperSize'], plan['helperSha256'])
        helper_manifest = plain_path(helper.parent / 'helper-files.json', root)
        if hashlib.sha256(helper_manifest.read_bytes()).hexdigest() != plan['helperFilesSha256']:
            fail('APPLY_HASH', 'helper 文件登记已变化。')
        helper_files = read_json(helper_manifest)
        if helper_files.get('schema') != 'ptb-update-helper/1' or not isinstance(helper_files.get('files'), list):
            fail('APPLY_METADATA', 'helper 文件登记无效。')
        for item in helper_files['files']:
            member = item.get('path')
            if not isinstance(member, str) or '\\' in member or ':' in member or any(p in ('', '.', '..') for p in member.split('/')):
                fail('APPLY_PATH', 'helper 文件路径无效。')
            verify_bytes(plain_path(helper.parent / member, root), item.get('size'), item.get('sha256'))
        closed = read_json(plain_path(request.parent / 'closed.json', root))
        if closed.get('schema') != 'ptb-update-closed/1' or closed.get('token') != token:
            fail('APPLY_CLOSE', '原程序未确认关闭。')
        _atomic_json(status, {'state': 'waiting', 'version': plan['version']})
        wait(handle, plan['parentPid'])
        package = plain_path(plan['package'], root / 'downloads')
        verify_bytes(package, plan['size'], plan['sha256'])
        origin = Path(plan['originExe'])
        if package_kind(origin) != plan['kind'] or SemVer(plan['version']) <= SemVer(plan['currentVersion']):
            fail('APPLY_METADATA', '换版类型或版本无效。')
        if plan['kind'] == 'portable':
            destination = origin.parent.parent / f'PhoneticToolbox-{plan["version"]}-{request.parent.name[:8]}'
            plain_path(destination, origin.parent.parent)
            exe = extract_portable(package, destination, plan['version'])
        elif plan['kind'] == 'installer':
            if not executable_magic(package):
                fail('APPLY_LAYOUT', '安装程序格式无效。')
            installer = launch([str(package), '/CURRENTUSER', '/VERYSILENT', '/SUPPRESSMSGBOXES', '/SP-', '/NORESTART', '/DIR=' + str(origin.parent)], cwd=str(package.parent))
            if installer.wait() != 0:
                fail('INSTALL_FAILED', '安装未成功完成，下载包与用户数据保留。')
            exe = release_layout(origin.parent, plan['version'])
        else:
            fail('APPLY_METADATA', '更新类型无效。')
        _atomic_json(status, {'state': 'launching', 'version': plan['version']})
        child = launch([str(exe)], cwd=str(exe.parent))
        _atomic_json(status, {'state': 'started', 'version': plan['version'], 'pid': child.pid,
                              'message': '新程序已启动；旧版本目录和更新包保留。'})
        return 0
    except Exception as error:
        _atomic_json(status, {'state': 'failed', 'code': getattr(error, 'code', 'APPLY_FAILED'),
                              'message': str(error) if isinstance(error, UpdateError) else '换版失败，原程序和更新包保留。'})
        return 1


def helper_main(args):
    if sys.platform != 'win32' or not getattr(sys, 'frozen', False) or len(args) != 3 or not args[2].isdigit():
        return 2
    handle = int(args[2])
    try:
        return run_handoff(Path(args[0]), args[1], handle)
    finally:
        kernel().CloseHandle(handle)
