"""Read-only font inspection in the same bounded EGG compatibility runtime."""
import json
import os
import queue
import subprocess
import sys
import threading
from .spectrogram_preview import PreviewError

_slot=threading.BoundedSemaphore(1)


def inspect_fonts(snapshot, *, timeout=20):
    from ptb_api.font_models import FigureFontSnapshot,FontPreflight
    from .egg_runtime import command
    from .acoustic_errors import AcousticFailure
    snapshot=FigureFontSnapshot.model_validate(snapshot)
    if sys.platform == 'linux':
        return _inspect_linux(snapshot, timeout)
    if os.name!='nt':raise PreviewError('font_preflight_platform_unverified',503)
    if not _slot.acquire(False):raise PreviewError('font_preflight_busy',429)
    from .native import windows as win
    process=job=None
    try:
        # Validate the fixed runtime before allocating any process handles.
        argv=command('--font-preflight','')[:-1]
        job=win.checked(win.create_job(None,None));limits=win.ExtendedLimit()
        limits.basic.flags=0x2000|0x100|0x200
        limits.process_memory=limits.job_memory=1_000_000_000
        win.checked(win.set_job(job,9,win.c.byref(limits),win.c.sizeof(limits)))
        process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        ready=queue.Queue()
        threading.Thread(target=lambda:ready.put(process.stdout.readline(64)),daemon=True).start()
        handle=win.checked(win.open_process(0x0101,False,int(ready.get(timeout=5))))
        try:win.checked(win.assign_job(job,handle))
        finally:win.close(handle)
        try:output,_=process.communicate(snapshot.model_dump_json().encode()+b'\n',timeout=timeout)
        except subprocess.TimeoutExpired:raise PreviewError('font_preflight_timeout',503) from None
        if process.returncode or len(output)>8192:raise PreviewError('font_preflight_failed',503)
        result=json.loads(output)
        if 'error' in result:raise PreviewError(result['error'],503)
        return FontPreflight.model_validate(result).model_dump()
    except PreviewError:raise
    except AcousticFailure as exc:raise PreviewError(exc.code,503) from None
    except (OSError,ValueError,queue.Empty):raise PreviewError('font_preflight_failed',503) from None
    finally:
        try:
            if job:win.terminate_job(job,1);win.close(job)
            if process:
                if process.poll() is None:process.terminate()
                process.wait(timeout=5)
                process.stdin.close();process.stdout.close()
        finally:_slot.release()


def _inspect_linux(snapshot, timeout):
    from pathlib import Path
    from ptb_api.font_models import FontPreflight
    from .native.linux_runtime import command, load_profile
    from .native.posix import run_bounded
    from .io.limits import Limits, LimitError, FormatError
    if not _slot.acquire(False):
        raise PreviewError('font_preflight_busy',429)
    try:
        _, profile = load_profile()
        raw = run_bounded(command('fonts', '--font-preflight'), snapshot.model_dump_json().encode()+b'\n',
                          Path(profile['cache']), Limits(input_bytes=8192, output_bytes=8192,
                          process_bytes=1073741824, timeout_seconds=timeout))
        return FontPreflight.model_validate_json(raw).model_dump()
    except LimitError as exc:
        raise PreviewError('font_preflight_timeout' if str(exc)=='native_timeout' else 'font_preflight_failed',503) from None
    except (OSError, ValueError, FormatError):
        raise PreviewError('font_preflight_failed',503) from None
    finally:
        _slot.release()


def child():
    print(os.getpid(),flush=True)
    try:
        from ptb_api.font_models import FigureFontSnapshot
        # Parent assigns the Job before releasing the child to import Matplotlib.
        snapshot=FigureFontSnapshot.model_validate_json(sys.stdin.buffer.readline(8192))
        from .egg_runtime import fingerprint
        fingerprint()
        from .fonts import check_fonts
        result=check_fonts(snapshot)
    except Exception as exc:
        from .acoustic_errors import public_error
        result={'error':public_error(exc)}
    sys.stdout.buffer.write(json.dumps(result,ensure_ascii=False,allow_nan=False).encode())
