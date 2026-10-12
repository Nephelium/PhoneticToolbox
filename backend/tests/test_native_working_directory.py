"""Real Win32 child launches preserve relative I/O in deep owned directories."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from scripts.verification_artifacts import verification_run

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend/src'))

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Win32 working directory entries')


def deep_directory(scratch):
    root = scratch / '中文任务目录'
    while len(str(root)) < 285:
        root /= 'long-working-directory'
    root.mkdir(parents=True)
    return root


def wait_for_exit(process):
    deadline = time.monotonic() + 10
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(.02)
    assert process.poll() == 0


def test_application_path_extends_long_drive_and_unc_paths_only():
    from ptb_worker.native.windows import application_path

    drive = 'D:\\' + ('native-directory\\' * 20) + 'worker.exe'
    unc = '\\\\server\\share\\' + ('native-directory\\' * 20) + 'worker.exe'
    assert application_path(drive) == '\\\\?\\' + drive
    assert application_path(unc) == '\\\\?\\UNC\\' + unc[2:]
    assert application_path('worker.exe') == 'worker.exe'
    assert application_path('D:\\short\\worker.exe') == 'D:\\short\\worker.exe'
    assert application_path('\\\\?\\' + drive) == '\\\\?\\' + drive


def test_long_working_directory_preserves_relative_io_and_cleans_only_entry():
    from ptb_worker.native.windows import OwnedProcess

    recipe = 'CPython -I -B child reads and writes relative files in a Chinese directory beyond MAX_PATH'
    with verification_run('native-paths', 'cwd-success', recipe) as (out, scratch):
        root = deep_directory(scratch)
        (scratch / 'unrelated-keep.txt').write_text('keep sibling', encoding='utf-8')
        (root / 'relative-input.txt').write_text('original input', encoding='utf-8')
        code = (
            'import json,os;from pathlib import Path;'
            'Path("cwd-result.json").write_text(json.dumps(dict(cwd=os.getcwd(),'
            'payload=Path("relative-input.txt").read_text(encoding="utf-8"))),encoding="utf-8")'
        )
        process = OwnedProcess([sys.executable, '-I', '-B', '-c', code], root, 268_435_456)
        alias = process.working_directory.path
        private_parent = alias.parent
        try:
            assert process.working_directory.owned
            assert str(alias).isascii() and len(str(alias)) < 140
            assert alias.samefile(root)
            wait_for_exit(process)
            result = json.loads((root / 'cwd-result.json').read_text(encoding='utf-8'))
            assert result['payload'] == 'original input'
            assert Path(result['cwd']).samefile(root)
            (out / 'report.json').write_text(json.dumps(dict(
                original_length=len(str(root)), entry_length=len(str(alias)), relative_io=True,
            )), encoding='utf-8')
        finally:
            process.close()
        assert process.group_cleaned
        assert not alias.exists() and not private_parent.exists()
        assert (root / 'relative-input.txt').read_text(encoding='utf-8') == 'original input'
        assert (root / 'cwd-result.json').is_file()
        assert (scratch / 'unrelated-keep.txt').read_text(encoding='utf-8') == 'keep sibling'
    assert json.loads((out / 'cleanup.json').read_text(encoding='utf-8-sig'))['status'] == 'deleted'


def test_failed_launch_releases_private_working_directory(monkeypatch):
    from ptb_worker import native_runtime as runtime
    from ptb_worker.native.windows import OwnedProcess

    recipe = 'Missing executable fails a Win32 launch after private deep-directory entry creation'
    with verification_run('native-paths', 'cwd-failure', recipe) as (out, scratch):
        root = deep_directory(scratch)
        (root / 'original-keep.txt').write_text('original input', encoding='utf-8')
        created = []
        create_private = runtime._private_directory

        def observe_private(base):
            parent = create_private(base)
            created.append(parent)
            return parent

        before = set(runtime._working_entries)
        monkeypatch.setattr(runtime, '_private_directory', observe_private)
        with pytest.raises(OSError):
            OwnedProcess([str(scratch / 'missing-executable.exe')], root, 268_435_456)
        assert created
        assert all(not parent.exists() for parent in created)
        assert set(runtime._working_entries) == before
        assert (root / 'original-keep.txt').read_text(encoding='utf-8') == 'original input'
        (out / 'report.json').write_text(json.dumps(dict(
            original_length=len(str(root)), created_entries=len(created), cleaned_entries=True,
        )), encoding='utf-8')
    assert json.loads((out / 'cleanup.json').read_text(encoding='utf-8-sig'))['status'] == 'deleted'


def frozen_fixture(scratch):
    root = deep_directory(scratch)
    original = root / 'PhoneticToolbox.exe'
    payload = b'MZ owned executable fixture\r\n' + bytes(range(256))
    original.write_bytes(payload)
    internal = root / '_internal'
    internal.mkdir()
    resource = internal / 'resource.txt'
    resource.write_text('original bundled resource', encoding='utf-8')
    return original, payload, resource


def test_frozen_executable_hard_link_keeps_canonical_entry_short(monkeypatch):
    from ptb_worker import native_runtime as runtime

    recipe = 'Tiny executable fixture has a short canonical hard-link entry and an internal resource junction'
    with verification_run('native-paths', 'frozen-hard-link', recipe) as (out, scratch):
        original, payload, resource = frozen_fixture(scratch)
        # The fixture workspace and ordinary TEMP may be on different volumes.
        # Select this existing owned ASCII base to exercise a real hard link.
        monkeypatch.setattr(runtime.tempfile, 'gettempdir', lambda: str(ROOT / 'output/validation'))
        entry = runtime.FrozenApplicationEntry(original)
        private_parent = entry.path.parent
        try:
            assert entry.path.samefile(original)
            assert entry.path.resolve() != original.resolve()
            assert str(entry.path.resolve()).isascii() and len(str(entry.path.resolve())) < 140
            assert (private_parent / '_internal/resource.txt').samefile(resource)
            assert entry.path.read_bytes() == payload
            (out / 'report.json').write_text(json.dumps(dict(
                original_length=len(str(original)), canonical_length=len(str(entry.path.resolve())),
                same_executable=True, same_resource=True,
            )), encoding='utf-8')
        finally:
            entry.close()
        assert not private_parent.exists()
        assert original.read_bytes() == payload
        assert resource.read_text(encoding='utf-8') == 'original bundled resource'
    assert json.loads((out / 'cleanup.json').read_text(encoding='utf-8-sig'))['status'] == 'deleted'


def test_frozen_executable_cross_volume_fallback_preserves_bytes_and_cleans_entry(monkeypatch):
    import errno
    from ptb_worker import native_runtime as runtime

    recipe = 'Simulated cross-volume hard-link rejection verifies copy bytes and preserves original executable/resources'
    with verification_run('native-paths', 'frozen-copy', recipe) as (out, scratch):
        original, payload, resource = frozen_fixture(scratch)

        def cross_volume(source, destination):
            raise OSError(errno.EXDEV, 'cross-volume hard link')

        monkeypatch.setattr(runtime.os, 'link', cross_volume)
        entry = runtime.FrozenApplicationEntry(original)
        private_parent = entry.path.parent
        try:
            assert not entry.path.samefile(original)
            assert entry.path.resolve() != original.resolve()
            assert str(entry.path.resolve()).isascii() and len(str(entry.path.resolve())) < 140
            assert entry.path.read_bytes() == payload
            assert (private_parent / '_internal/resource.txt').samefile(resource)
            (out / 'report.json').write_text(json.dumps(dict(
                original_length=len(str(original)), canonical_length=len(str(entry.path.resolve())),
                copied_bytes=len(payload), same_resource=True,
            )), encoding='utf-8')
        finally:
            entry.close()
        assert not private_parent.exists()
        assert original.read_bytes() == payload
        assert resource.read_text(encoding='utf-8') == 'original bundled resource'
    assert json.loads((out / 'cleanup.json').read_text(encoding='utf-8-sig'))['status'] == 'deleted'


@pytest.mark.parametrize('mode', ['hardlink', 'copy', 'corrupt', 'reparse'])
def test_guarded_frozen_entry_mirrors_plain_resources_and_cleans_on_failure(mode):
    recipe = 'Real child RedirectionTrust enforcement rejects junctions; plain frozen mirror preserves bytes and closes only its own resources'
    with verification_run('native-paths', 'frozen-guard-' + mode, recipe) as (out, scratch):
        original, payload, resource = frozen_fixture(scratch)
        report_path = out / 'report.json'
        code = r'''
import ctypes, errno, hashlib, json, os, shutil, stat, sys
from ctypes import wintypes
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from ptb_worker import native_runtime as runtime
from ptb_worker.io.scratch import no_links
original, report_path, base = map(Path, sys.argv[2:5])
mode = sys.argv[5]
api = ctypes.WinDLL('kernel32', use_last_error=True)
api.SetProcessMitigationPolicy.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
api.SetProcessMitigationPolicy.restype = wintypes.BOOL
api.GetProcessMitigationPolicy.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
api.GetProcessMitigationPolicy.restype = wintypes.BOOL
api.GetCurrentProcess.restype = wintypes.HANDLE
value = wintypes.DWORD(1)
assert api.SetProcessMitigationPolicy(16, ctypes.byref(value), ctypes.sizeof(value))
assert api.GetProcessMitigationPolicy(api.GetCurrentProcess(), 16, ctypes.byref(value), ctypes.sizeof(value)) and value.value == 1
parents = []
def private_parent(**options):
    no_links(base)
    parent = runtime._private_directory(base)
    parents.append(parent)
    return parent
runtime._private_parent = private_parent
if mode in ('copy', 'corrupt'):
    def cross_volume(*args): raise OSError(errno.EXDEV, 'simulated different volume')
    runtime.os.link = cross_volume
if mode == 'corrupt':
    def corrupt(source, target): Path(target).write_bytes(b'wrong bytes')
    shutil.copyfile = corrupt
bad = original.parent / '_internal/bad'
if mode == 'reparse':
    import _winapi
    _winapi.CreateJunction(str(original.parent), str(bad))
entry = None
before = set(runtime._working_entries)
try:
    if mode in ('corrupt', 'reparse'):
        try: runtime.FrozenApplicationEntry(original)
        except ValueError: pass
        else: raise AssertionError('Invalid resource/copy was accepted')
    else:
        entry = runtime.FrozenApplicationEntry(original)
        assert entry.link is None and entry.mirror is not None
        assert str(entry.path.resolve()).isascii() and len(str(entry.path.resolve())) < 140
        no_links(entry.path)
        mirrored_resource = entry.path.parent / '_internal/resource.txt'
        assert mirrored_resource.read_text('utf8') == 'original bundled resource'
        assert entry.path.read_bytes() == original.read_bytes()
        assert entry.path.samefile(original) is (mode == 'hardlink')
        assert mirrored_resource.samefile(original.parent / '_internal/resource.txt') is (mode == 'hardlink')
        parent = entry.path.parent
        entry.close()
        assert not parent.exists()
    assert set(runtime._working_entries) == before
    assert all(not parent.exists() for parent in parents)
    assert original.read_bytes().startswith(b'MZ owned executable fixture')
    assert (original.parent / '_internal/resource.txt').read_text('utf8') == 'original bundled resource'
    assert api.GetProcessMitigationPolicy(api.GetCurrentProcess(), 16, ctypes.byref(value), ctypes.sizeof(value)) and value.value == 1
    report_path.write_text(json.dumps({'success': True, 'mode': mode, 'guardFlags': value.value, 'cleanedParents': len(parents), 'originalSha256': hashlib.sha256(original.read_bytes()).hexdigest()}), 'utf8')
finally:
    if entry: entry.close()
    if mode == 'reparse': os.rmdir(bad)
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-X', 'utf8', '-c', code,
                                 str(ROOT / 'backend/src'), str(original), str(report_path),
                                 str(ROOT / 'output/validation'), mode],
                                capture_output=True, text=True, encoding='utf8', timeout=30,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(report_path.read_text('utf8'))['guardFlags'] == 1
        assert original.read_bytes() == payload
        assert resource.read_text('utf8') == 'original bundled resource'
    assert json.loads((out / 'cleanup.json').read_text(encoding='utf-8-sig'))['status'] == 'deleted'


@pytest.mark.parametrize('deep_binary', [False, True], ids=['short-exe', 'long-chinese-exe'])
def test_real_reaper_in_long_chinese_scratch_uses_relative_input_and_preserves_validation(monkeypatch, deep_binary):
    from dataclasses import replace
    import numpy as np
    from phonetic_core.models.audio import AudioInput
    from ptb_worker.io.limits import Limits
    from ptb_worker.io.scratch import Scratch, no_links
    from ptb_worker.native import reaper
    from ptb_worker.native import windows
    from ptb_worker.native.windows import close, open_process, wait

    recipe = 'Registered real REAPER analyzes a 150 Hz harmonic tone in Chinese scratch beyond MAX_PATH'
    with verification_run('native-paths', 'reaper-long', recipe) as (out, scratch):
        root = deep_directory(scratch)
        binary = ROOT / 'phonetic_toolbox/core/acoustic/reaper.exe'
        if deep_binary:
            binary = root / '中文原生程序' / 'reaper.exe'
            binary.parent.mkdir()
            shutil.copyfile(ROOT / 'phonetic_toolbox/core/acoustic/reaper.exe', binary)
            assert len(str(binary)) > 260
        no_links(binary)
        calls = []
        applications = []
        collect = reaper.collect_pipe
        create = windows.create_process

        def observe_application(application, command, *args):
            applications.append(dict(application=application, command=command.value))
            return create(application, command, *args)

        def observe(argv, pipe, cwd, *args, **kwargs):
            input_name = argv[argv.index('-i') + 1]
            calls.append(dict(binary=argv[0], input=input_name, cwd=str(cwd)))
            # Inspect the genuinely created input before the real native launch.
            assert Path(cwd, input_name).is_file()
            return collect(argv, pipe, cwd, *args, **kwargs)

        monkeypatch.setattr(reaper, 'collect_pipe', observe)
        monkeypatch.setattr(windows, 'create_process', observe_application)
        phase = 2 * np.pi * 150 * np.arange(16000) / 16000
        audio = AudioInput(sum(np.sin(k * phase) / (k*k) for k in range(1, 13)) * .4, 16000)
        with Scratch(root, 1_000_000) as capability:
            native = reaper.Reaper(binary, capability, replace(Limits(), timeout_seconds=10))
            assert native.sha256 == reaper.REAPER_SHA256
            result = native(audio, .005, 60, 880, hilbert=True, no_highpass=False)
            assert result.actual_backend == 'native_reaper' and len(result.times) > 0
            np.testing.assert_allclose(np.diff(result.times), .005, rtol=0, atol=1e-12)
            median = float(np.nanmedian(result.values))
            assert abs(median - 150) < 2
            assert capability.used == 0 and not list(capability.root.iterdir())
            assert len(calls) == 1
            assert len(applications) == 1
            assert applications[0]['application'] == windows.application_path(binary)
            assert str(binary) in applications[0]['command']
            assert calls[0]['input'].isascii() and Path(calls[0]['input']).name == calls[0]['input']
            assert not Path(calls[0]['input']).is_absolute()
            assert len(calls[0]['cwd']) > 260
            handle = open_process(0x100000, False, native.last_pid)
            try:
                assert not handle or wait(handle, 0) == 0
            finally:
                if handle: close(handle)
            (out / 'report.json').write_text(json.dumps(dict(
                actual_backend=result.actual_backend, median_f0_hz=median, frames=len(result.times),
                binary_length=len(str(binary)), scratch_length=len(calls[0]['cwd']),
                input_argument=calls[0]['input'], binary_sha256=native.sha256, cleaned_process=True,
                application_argument=applications[0]['application'],
            )), encoding='utf-8')
        assert binary.is_file()
    assert json.loads((out / 'cleanup.json').read_text(encoding='utf-8-sig'))['status'] == 'deleted'


def test_reaper_long_path_keeps_junction_and_sha_rejection():
    from ptb_worker.io.scratch import Scratch
    from ptb_worker.native.reaper import Reaper
    from ptb_worker.native_runtime import DirectoryEntry

    recipe = 'Real registered executable keeps strict junction and changed-byte rejection in a deep directory'
    with verification_run('native-paths', 'reaper-validation', recipe) as (out, scratch):
        root = deep_directory(scratch)
        binary = root / 'reaper.exe'
        shutil.copyfile(ROOT / 'phonetic_toolbox/core/acoustic/reaper.exe', binary)
        with Scratch(root, 1_000_000) as capability:
            entry = DirectoryEntry(root)
            try:
                assert entry.owned
                with pytest.raises(ValueError, match='Reparse'):
                    Reaper(entry.path / binary.name, capability)
            finally:
                entry.close()
            changed = bytearray(binary.read_bytes()); changed[-1] ^= 1
            binary.write_bytes(changed)
            with pytest.raises(ValueError, match='Unregistered native binary'):
                Reaper(binary, capability)
        (out / 'report.json').write_text(json.dumps(dict(
            junction_rejected=True, changed_sha_rejected=True,
        )), encoding='utf-8')
    assert json.loads((out / 'cleanup.json').read_text(encoding='utf-8-sig'))['status'] == 'deleted'
