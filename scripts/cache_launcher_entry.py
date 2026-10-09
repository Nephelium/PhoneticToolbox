"""Small bootstrap; the frozen application remains the sole business entry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import time

from ptb_desktop.startup_cache import Payload, cache_root, clear_if_idle, prepare, maintenance, active_leases, request_clear
from ptb_desktop.startup_progress import PreparationWindow
from ptb_desktop.startup_ready import ReadyEvent, ENVIRONMENT


def main():
    arguments = sys.argv[1:]
    uninstall = arguments == ['--ptb-clear-all-caches']
    if uninstall and cache_root().exists():
        with maintenance(cache_root()):
            if active_leases(cache_root()): return 21
    if arguments == ['--ptb-clear-startup-cache']:
        return 0 if clear_if_idle()['complete'] else 21
    progress = PreparationWindow(); live = None; ready = None
    if os.environ.get('PTB_OWNED_BOOTSTRAP_LOG') and os.environ.get('PTB_OWNED_STARTUP_CARD')=='offscreen':
        progress.offscreen=True;progress.quiet=False
    try:
        progress.update('starting',0,0)
        config = json.loads((Path(sys._MEIPASS) / 'cache-payload.json').read_text('utf8'))
        payload = Payload(sys.executable, config)
        from ptb_desktop.cache_cleanup import disposable_caches
        clear_if_idle(requested_only=True,extra_cleanup=disposable_caches)
        application, live, report = prepare(payload, progress=progress.update)
        if arguments == ['--ptb-prepare-cache']: return 0
        env = os.environ.copy()
        for name in list(env):
            if name.startswith('_PYI_'): env.pop(name, None)
        env.update(PYINSTALLER_RESET_ENVIRONMENT='1', PYTHONDONTWRITEBYTECODE='1',
                   PTB_DISTRIBUTION_EXE=str(Path(sys.executable).absolute()),
                   PTB_LAUNCHER_PID=str(os.getpid()))
        # Only an ordinary GUI launch waits for workbench readiness. Workers,
        # update helpers and command-line verification do not construct a GUI.
        if not arguments:
            ready=ReadyEvent();env[ENVIRONMENT]=ready.name
            progress.update('loading',0,0)
        else:
            env.pop(ENVIRONMENT,None);progress.close()
        # A bootloader's temporary DLL directory must not leak into the app.
        import ctypes
        ctypes.windll.kernel32.SetDllDirectoryW(None)
        if uninstall: arguments=['--ptb-clear-user-caches']
        options={}
        if len(arguments)==4 and arguments[0]=='--ptb-apply-update' and arguments[-1].isdigit():
            # The updater's verified original-process handle crosses both the
            # small launcher and its frozen application, without PID polling.
            from ctypes import wintypes
            handle=int(arguments[-1])
            api=ctypes.WinDLL('kernel32',use_last_error=True)
            api.SetHandleInformation.argtypes=(wintypes.HANDLE,wintypes.DWORD,wintypes.DWORD)
            if not api.SetHandleInformation(handle,1,1):raise ctypes.WinError(ctypes.get_last_error())
            startup=subprocess.STARTUPINFO();startup.lpAttributeList={'handle_list':[handle]}
            options.update(startupinfo=startup,close_fds=True)
        child = subprocess.Popen([str(application), *arguments], env=env, cwd=str(Path(sys.executable).parent), **options)
        if ready:
            started=time.monotonic();slow=False
            while child.poll() is None and not ready.is_set():
                if not slow and time.monotonic()-started>=30:
                    progress.update('slow-loading',0,0);slow=True
                time.sleep(.05)
            progress.close();ready.close();ready=None
        code=child.wait()
        if uninstall and code==0:
            request_clear();live.close();live=None
            from ptb_desktop.cache_cleanup import disposable_caches
            return 0 if clear_if_idle(extra_cleanup=disposable_caches)['complete'] else 21
        return code
    finally:
        progress.close()
        if ready:ready.close()
        if live: live.close()
        from ptb_desktop.cache_cleanup import disposable_caches
        clear_if_idle(requested_only=True,extra_cleanup=disposable_caches)


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as error:
        log = os.environ.get('PTB_OWNED_BOOTSTRAP_LOG')
        path = Path(log) if log else cache_root().parent / 'startup-error.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(traceback.format_exc(), 'utf8')
        if not log:
            import ctypes
            message=str(error)
            if not any('\u4e00'<=character<='\u9fff' for character in message):
                message='文件完整性或目录访问检查未通过，请检查磁盘空间、文件权限或重新下载应用。'
            ctypes.windll.user32.MessageBoxW(None, '运行文件未能准备完成。\n' + message + '\n\n请关闭其他窗口后重试。已有研究资料保留。', 'PhoneticToolbox 启动未完成', 0x10)
        raise SystemExit(1)
