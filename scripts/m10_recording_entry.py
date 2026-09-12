"""Local M10 recording artifact; no production or database migration entry."""
import os
import sys
from pathlib import Path


def restore_worker_pipes():
    # PyInstaller windowed executables set Python streams to None even when a
    # parent explicitly supplies pipes. Recover only this child's inherited IO.
    import ctypes
    import msvcrt
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetStdHandle.argtypes=[ctypes.c_ulong];kernel.GetStdHandle.restype=ctypes.c_void_p
    for name,number,mode in [('stdin',-10,'r'),('stdout',-11,'w'),('stderr',-12,'w')]:
        existing=getattr(sys,name)
        if existing is not None:
            if hasattr(existing,'reconfigure'):existing.reconfigure(encoding='utf-8',errors='strict')
            continue
        handle=kernel.GetStdHandle(number & 0xffffffff)
        if handle and handle!=ctypes.c_void_p(-1).value:
            fd=msvcrt.open_osfhandle(handle,os.O_RDONLY if mode=='r' else os.O_WRONLY)
            stream=os.fdopen(fd,mode,encoding='utf-8',buffering=1)
        else:stream=open(os.devnull,mode,encoding='utf-8')
        setattr(sys,name,stream)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--ptb-worker':
        restore_worker_pipes()
        from ptb_worker.process_entry import MODULES, dispatch
        if len(sys.argv) < 3 or sys.argv[2] not in MODULES: return 2
        dispatch(sys.argv[2], sys.argv[3:]);return 0
    if '--m10-worker' in sys.argv:
        restore_worker_pipes()
        from ptb_desktop.vocal_tract.worker import main as worker
        worker();return 0
    if '--local-service' in sys.argv:
        restore_worker_pipes();sys.argv.remove('--local-service')
        from ptb_api.cli import main as service
        service();return 0
    root=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[1]))
    if '--self-test-r4' in sys.argv:
        restore_worker_pipes()
        import verify_m10_recording_features
        verify_m10_recording_features.main(root=root,out=Path(sys.argv[sys.argv.index('--self-test-r4')+1]));return 0
    if '--self-test' in sys.argv:
        restore_worker_pipes()
        import verify_m10_qt
        verify_m10_qt.main(root=root,out=Path(sys.argv[sys.argv.index('--self-test')+1]));return 0
    if any(arg.startswith('-') for arg in sys.argv[1:]):
        return 2  # Unknown child switches must never reopen the GUI.
    from ptb_desktop.host import run
    return run(root/'frontend/dist',start_module='M10',vocal_resources=root/'resources/vocal_tract/native')


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    test_flag=next((flag for flag in ['--self-test','--self-test-r4'] if flag in sys.argv),None)
    if test_flag:
        try:raise SystemExit(main())
        except Exception:
            import traceback
            out=Path(sys.argv[sys.argv.index(test_flag)+1]);out.mkdir(parents=True,exist_ok=True)
            (out/'startup-error.log').write_text(traceback.format_exc(),encoding='utf-8')
            raise SystemExit(1)
    else:raise SystemExit(main())
