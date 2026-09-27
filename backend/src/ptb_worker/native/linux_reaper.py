"""REAPER output pipe inside the already limited scientific cgroup."""
import os
from pathlib import Path
import selectors
import subprocess
import time
from ..io.limits import Cancelled, FormatError, LimitError


def require_group():
    group = next((s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')), '')
    if not any(p.startswith('ptb-p11-') and p.endswith('.service') for p in Path(group).parts):
        raise FormatError('linux_process_boundary_unavailable')
    root = Path('/sys/fs/cgroup')/group.lstrip('/')
    memory = (root/'memory.max').read_text().strip()
    quota, period = (root/'cpu.max').read_text().split()
    from ..resource_profiles import selected_profile
    profile = selected_profile()
    if not memory.isdigit():
        raise FormatError('linux_process_boundary_unavailable')
    if profile.name == 'server-small' and (int(memory)>profile.memory_bytes or quota=='max' or int(quota)>int(period)):
        raise FormatError('linux_process_boundary_unavailable')


def collect(argv, cwd, limits, stop, on_started=None):
    require_group()
    reader, writer = os.pipe()
    process = None
    selector = selectors.DefaultSelector()
    started = time.monotonic()
    result = bytearray()
    try:
        args = list(argv)
        args[args.index('-f')+1] = '/proc/self/fd/'+str(writer)
        process = subprocess.Popen(args, cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, pass_fds=(writer,))
        os.close(writer);writer=None
        if on_started:on_started(process.pid)
        os.set_blocking(reader,False);selector.register(reader,selectors.EVENT_READ)
        while selector.get_map() or process.poll() is None:
            if stop():raise Cancelled('cancelled')
            if time.monotonic()-started>limits.timeout_seconds:raise LimitError('native_timeout')
            for _,_ in selector.select(.02):
                block=os.read(reader,min(65536,limits.output_bytes-len(result)+1))
                if not block:selector.unregister(reader)
                else:
                    result.extend(block)
                    if len(result)>limits.output_bytes:raise LimitError('output_bytes_exceeded')
        if process.wait(timeout=3)!=0:raise FormatError('native_reaper_failed')
        return bytes(result),process.pid
    finally:
        selector.close();os.close(reader)
        if writer is not None:os.close(writer)
        if process is not None and process.poll() is None:
            process.kill();process.wait(timeout=3)
