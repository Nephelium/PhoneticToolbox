"""Native REAPER pipe inside the macOS science process group."""
import os,selectors,subprocess,time
from .darwin import _memory
from ..io.limits import Cancelled,FormatError,LimitError

def collect(argv,cwd,limits,stop,on_started=None):
    reader,writer=os.pipe();process=None;selector=selectors.DefaultSelector();started=time.monotonic();result=bytearray()
    try:
        args=list(argv);args[args.index('-f')+1]='/dev/fd/'+str(writer)
        process=subprocess.Popen(args,cwd=cwd,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,pass_fds=(writer,))
        os.close(writer);writer=None
        if on_started:on_started(process.pid)
        os.set_blocking(reader,False);selector.register(reader,selectors.EVENT_READ)
        while selector.get_map() or process.poll() is None:
            if stop():raise Cancelled('cancelled')
            if time.monotonic()-started>limits.timeout_seconds:raise LimitError('native_timeout')
            if _memory(process)>limits.process_bytes:raise LimitError('native_memory_exceeded')
            for key,mask in selector.select(.02):
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
        if process is not None and process.poll() is None:process.kill();process.wait(timeout=3)
