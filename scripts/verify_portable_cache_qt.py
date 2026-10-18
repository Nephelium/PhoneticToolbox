"""Owned Win32 chooser and real Qt cache settings; isolated data, no release."""
import ctypes as c
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import threading
import time

from verification_artifacts import verification_run, write_json
from workbench_source import ROOT, configure


def native_choice(out, parent, *, cancel=False):
    """Operate only dialogs belonging to this verification process."""
    from ptb_desktop.cache_location_dialog import choose_directory
    user = c.WinDLL('user32', use_last_error=True)
    callback_type = c.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
    user.EnumWindows.argtypes = (callback_type, w.LPARAM)
    user.GetWindowThreadProcessId.argtypes = (w.HWND, c.POINTER(w.DWORD))
    user.GetWindowTextW.argtypes = (w.HWND, w.LPWSTR, c.c_int)
    user.GetClassNameW.argtypes = (w.HWND, w.LPWSTR, c.c_int)
    user.SendMessageW.argtypes = (w.HWND, w.UINT, w.WPARAM, w.LPARAM)
    user.SendMessageW.restype = w.LPARAM
    user.PostMessageW.argtypes = (w.HWND, w.UINT, w.WPARAM, w.LPARAM)
    user.IsWindowVisible.argtypes = (w.HWND,)
    errors = []
    def control():
        stage, deadline = 0, time.monotonic() + 25
        try:
            while time.monotonic() < deadline:
                dialogs = []
                @callback_type
                def collect(hwnd, _):
                    pid = w.DWORD()
                    user.GetWindowThreadProcessId(hwnd, c.byref(pid))
                    name, title = c.create_unicode_buffer(128), c.create_unicode_buffer(256)
                    user.GetClassNameW(hwnd, name, len(name))
                    user.GetWindowTextW(hwnd, title, len(title))
                    if pid.value == os.getpid() and user.IsWindowVisible(hwnd) and name.value == '#32770':
                        dialogs.append((hwnd, title.value))
                    return True
                user.EnumWindows(collect, 0)
                for hwnd, title in dialogs:
                    if stage == 0 and title == '选择缓存保存位置':
                        user.PostMessageW(hwnd, 0x111, 2 if cancel else 1, 0)
                        if cancel:
                            return
                        stage = 1
                    elif stage == 1 and title != '选择缓存保存位置':
                        selection = c.create_unicode_buffer(str(parent))
                        user.SendMessageW(hwnd, 0x400 + 103, 1, c.cast(selection, c.c_void_p).value)
                        time.sleep(.25)
                        user.PostMessageW(hwnd, 0x111, 1, 0)
                        return
                time.sleep(.05)
            raise TimeoutError('Owned folder chooser was not ready')
        except Exception as error:
            errors.append(str(error))
    worker = threading.Thread(target=control, daemon=True)
    worker.start()
    result = choose_directory()
    worker.join(5)
    assert not errors and not worker.is_alive(), errors
    assert result is None if cancel else result == parent, result
    return result


def main():
    configure()
    with verification_run('portable-cache-location', 'qt',
                          'Owned Win32 chooser; temporary profile/workspace; real Qt native bridge, cache checks, confirmation, theme and viewport') as (out, scratch):
        os.environ['LOCALAPPDATA'] = str(scratch / 'profile')
        os.environ['QT_QPA_PLATFORM'] = 'windows'
        os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--mute-audio'
        from ptb_desktop import cache_location as location
        os.environ.pop(location.ENVIRONMENT, None)
        parent = scratch / '井井的缓存 位置'
        parent.mkdir()
        assert native_choice(out, parent, cancel=True) is None
        selected = native_choice(out, parent)
        location.configure_launcher(scratch / 'PhoneticToolbox.exe', [], lambda reason: selected)
        print('Native cancellation and folder selection passed', flush=True)

        from ptb_worker.local_workspace import prepare_workspace
        database, files = prepare_workspace(scratch / 'workspace', ROOT / 'backend/migrations')
        from ptb_desktop.startup_cache import SCHEMA, lease
        old = location.configuration_path().parent / 'startup-cache'
        old.mkdir(parents=True)
        (old / 'owner.json').write_text(json.dumps({'schema': SCHEMA}), encoding='utf8')
        (old / 'apps').mkdir()
        (old / 'apps/previous.bin').write_bytes(b'previous-version-cache')
        protected = location.configuration_path().parent / 'papers/annotations.json'
        protected.parent.mkdir()
        protected.write_text('{"note":"keep"}', encoding='utf8')

        from unittest.mock import patch
        from PyQt6.QtCore import QEventLoop, QTimer, QEvent
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtWidgets import QApplication
        from ptb_desktop.host import Workbench, register_scheme
        register_scheme()
        app = QApplication(['portable-cache-owned-QA'])
        app.setApplicationName('PhoneticToolbox-v3')
        # Network maintenance is outside this local UI verification. Storage,
        # native bridges, local API and the Chromium profile remain real.
        with patch('ptb_desktop.papers_bridge.PapersBridge.start_maintenance'), patch('ptb_desktop.updates.UpdateService.start_maintenance'):
            window = Workbench(ROOT / 'frontend/dist', jobs_path=database, local_files_root=files,
                               vocal_profile=scratch / 'vocal')
        service = window.service.process
        opened = []
        def pause(ms=100):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()
        def js(code):
            loop, values = QEventLoop(), []
            window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
            QTimer.singleShot(10_000, loop.quit)
            loop.exec()
            assert values, 'Qt JavaScript timed out'
            return values[0]
        def until(code):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-2500)')))
        def click(text):
            until("(()=>{const b=[...document.querySelectorAll('button')].find(e=>e.offsetParent&&e.textContent.trim()==="+json.dumps(text)+");if(!b||b.disabled)return false;b.click();return true})()")
        try:
            window.resize(1280, 900)
            window.show()
            until("!!document.querySelector('button[title=\"设置\"]')")
            js("document.querySelector('button[title=\"设置\"]').click()")
            click('本机存储与自动清理')
            until("!!document.querySelector('.cache-location code')&&document.querySelectorAll('.old-cache-item[data-kind=runtime]').length===1")
            assert Path(js("document.querySelector('.cache-location code').textContent")) == location.portable_directory()
            assert window.profile.cachePath() == str(location.browser_directory())
            assert Path(window.profile.persistentStoragePath()) == location.windows_workbench_storage()
            with patch.object(QDesktopServices, 'openUrl', lambda url: opened.append(url.toLocalFile()) or True):
                click('打开缓存目录')
                js("document.querySelector('.old-cache-item[data-kind=runtime] button').click()")
            assert [Path(path) for path in opened] == [location.portable_directory(), old]
            click('清理这处缓存…')
            assert js("document.querySelector('[aria-label=\"确认清理缓存\"] button').disabled") is True
            pause(350)
            window.view.grab().save(str(out / 'old-cache-confirmation.png'))
            click('取消')
            assert (old / 'apps/previous.bin').exists()
            click('清理这处缓存…')
            js("document.querySelector('[aria-label=\"确认清理缓存\"] input').click()")
            live = lease(old)
            try:
                click('确认清理此处缓存')
                until("document.querySelector('.old-caches').innerText.includes('仍有程序')")
                assert (old / 'apps/previous.bin').exists()
            finally:
                live.close()
            click('确认清理此处缓存')
            until("document.querySelectorAll('.old-cache-item[data-kind=runtime]').length===0")
            assert not old.exists() and protected.read_text('utf8') == '{"note":"keep"}'
            for mode in ('light', 'dark'):
                js("document.documentElement.dataset.theme=" + json.dumps(mode))
                for width in (1280, 760):
                    window.resize(width, 900)
                    pause(350)
                    assert js("document.documentElement.scrollWidth<=innerWidth+1")
                    window.view.grab().save(str(out / f'storage-{mode}-{width}.png'))
            js("localStorage.setItem('cache-preservation-check','research-draft-kept')")
            until("document.querySelectorAll('.old-cache-item[data-kind=graphics]').length>0")
            graphics_path=Path(js("document.querySelector('.old-cache-item[data-kind=graphics] code').textContent"))
            js("document.querySelectorAll('.old-cache-item[data-kind=graphics] button')[1].click()")
            js("document.querySelector('[aria-label=\"确认清理缓存\"] input').click()")
            click('确认并退出清理')
            deadline=time.monotonic()+20
            while service.poll() is None and time.monotonic()<deadline:pause()
            assert service.poll()==0 and (location.user_data_root()/'graphics-cache-clear.json').exists()
            print('Real Qt location, confirmation, concurrent-use protection and safe graphics-only exit passed', flush=True)
        finally:
            window.closing = True
            window.close()
            window.page.deleteLater()
            window.profile.deleteLater()
            app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            pause(100)
        assert service.poll() == 0
        from ptb_desktop.graphics_cache import cleanup_requested
        assert cleanup_requested()['complete'] and not graphics_path.exists()
        with patch('ptb_desktop.papers_bridge.PapersBridge.start_maintenance'), patch('ptb_desktop.updates.UpdateService.start_maintenance'):
            window=Workbench(ROOT/'frontend/dist',jobs_path=database,local_files_root=files,vocal_profile=scratch/'vocal')
        try:
            window.show()
            until("localStorage.getItem('cache-preservation-check')==='research-draft-kept'")
            assert window.profile.cachePath()==str(location.browser_directory())
        finally:
            window.closing=True;window.close();window.page.deleteLater();window.profile.deleteLater()
            app.sendPostedEvents(None,QEvent.Type.DeferredDelete);pause()
        assert window.service.process.poll()==0
        write_json(out / 'report.json', dict(success=True, native_choice=True,
                   qt_cases=['path display','native directory requests','confirmation required','cancel preserves files',
                             'late active lease blocks cleanup','old cache cleanup preserves paper annotation',
                             'light/dark and 1280/760 layouts','graphics-only safe exit','real localStorage preserved on reopen'],
                   opened_paths=opened, user_data_retained=True,
                   cache_files=[str(path.relative_to(scratch)) for path in scratch.rglob('*') if path.is_file()],
                   final_exe_verified=False))
        print(json.dumps({'success': True, 'output': str(out)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
