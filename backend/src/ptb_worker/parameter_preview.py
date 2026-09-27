"""M02 read-only table decoding in an owned bounded child. No disk output."""
import hashlib
from ptb_worker.process_entry import command
import json
import os
import queue
import subprocess
import sys
import threading
from .spectrogram_preview import PreviewError

MAX_BYTES = 16_000_000


def render(raw, name, timeout=20):
    if not 0 < len(raw) <= MAX_BYTES:
        raise PreviewError('parameter_input_budget', 413)
    if sys.platform == 'linux':
        return _render_linux(raw, name, timeout)
    if os.name != 'nt':
        raise PreviewError('preview_platform_unverified', 503)
    from .native import windows as win
    process = job = None
    try:
        job = win.checked(win.create_job(None, None))
        limits = win.ExtendedLimit()
        limits.basic.flags = 0x2000 | 0x100 | 0x200
        limits.process_memory = limits.job_memory = 512_000_000
        win.checked(win.set_job(job, 9, win.c.byref(limits), win.c.sizeof(limits)))
        process = subprocess.Popen(command('ptb_worker.parameter_preview'),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW)
        ready = queue.Queue()
        threading.Thread(target=lambda: ready.put(process.stdout.readline(64)), daemon=True).start()
        handle = win.checked(win.open_process(0x0101, False, int(ready.get(timeout=5))))
        try:
            win.checked(win.assign_job(job, handle))
        finally:
            win.close(handle)
        header = json.dumps({'size': len(raw), 'name': name}).encode() + b'\n'
        try:
            output, _ = process.communicate(header + raw, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise PreviewError('parameter_read_timeout', 503) from None
        if process.returncode or len(output) > MAX_BYTES:
            raise PreviewError('parameter_read_failed')
        value = json.loads(output)
        if 'error' in value:
            raise PreviewError('invalid_parameter_table')
        from ptb_api.display_models import ParameterTable
        return ParameterTable.model_validate(value).model_dump()
    except PreviewError:
        raise
    except (ValueError, OSError, queue.Empty):
        raise PreviewError('parameter_read_failed') from None
    finally:
        if job:
            win.terminate_job(job, 1)
            win.close(job)
        if process:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()


def _render_linux(raw, name, timeout):
    from pathlib import Path
    from .native.process import run_fixed_module
    from .io.limits import Limits, FormatError, LimitError
    from ptb_api.display_models import ParameterTable
    header = json.dumps({'size': len(raw), 'name': name}).encode() + b'\n'
    if len(header) > 8192:
        raise PreviewError('parameter_input_budget', 413)
    try:
        output = run_fixed_module('ptb_worker.parameter_preview', header + raw, Path(sys.prefix),
                                  Limits(input_bytes=MAX_BYTES + 8192, output_bytes=MAX_BYTES + 64,
                                         process_bytes=512_000_000, timeout_seconds=timeout))
        pid, separator, encoded = output.partition(b'\n')
        if not separator or not pid.isdigit() or len(pid) > 20 or len(encoded) > MAX_BYTES:
            raise ValueError('Invalid child protocol')
        value = json.loads(encoded)
        if 'error' in value:
            raise PreviewError('invalid_parameter_table')
        return ParameterTable.model_validate(value).model_dump()
    except PreviewError:
        raise
    except LimitError as exc:
        raise PreviewError('parameter_read_timeout' if str(exc) == 'native_timeout' else 'parameter_read_failed', 503) from None
    except FormatError as exc:
        raise PreviewError('preview_platform_unverified' if str(exc) == 'linux_process_boundary_unavailable' else 'parameter_read_failed', 503) from None
    except (OSError, ValueError, RuntimeError):
        raise PreviewError('parameter_read_failed') from None


def child():
    print(os.getpid(), flush=True)
    try:
        header = json.loads(sys.stdin.buffer.readline(8192))
        size = header['size']
        if type(size) != int or not 0 < size <= MAX_BYTES:
            raise ValueError('Input budget')
        raw = sys.stdin.buffer.read(size)
        if len(raw) != size:
            raise ValueError('Incomplete input')
        from .io.legacy_parameters import read_legacy_parameters
        from ptb_api.display_models import ParameterTable
        table = read_legacy_parameters(raw, header['name'])
        result = ParameterTable(sha256=hashlib.sha256(raw).hexdigest(), **table).model_dump()
    except Exception:
        result = {'error': 'invalid_parameter_table'}
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(encoded) > MAX_BYTES:
        raise SystemExit(2)
    sys.stdout.buffer.write(encoded)


if __name__ == '__main__':
    child()
