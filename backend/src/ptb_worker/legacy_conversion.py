"""Local-only, memory-only PKL conversion in an owned bounded Windows Job."""
import json
from ptb_worker.process_entry import command
import os
import queue
import struct
import subprocess
import sys
import threading

MAX_INPUT=18_000_008
MAX_OUTPUT=2_000_000


def convert(payload,timeout=20):
    from .spectrogram_preview import PreviewError
    if not 8<len(payload)<=MAX_INPUT:raise PreviewError('legacy_conversion_budget',413)
    if os.name!='nt':raise PreviewError('legacy_conversion_platform_unverified',503)
    from .native import windows as win
    process=None;job=None
    try:
        job=win.checked(win.create_job(None,None));limits=win.ExtendedLimit()
        limits.basic.flags=0x2000|0x100|0x200
        limits.process_memory=512_000_000;limits.job_memory=512_000_000
        win.checked(win.set_job(job,9,win.c.byref(limits),win.c.sizeof(limits)))
        process=subprocess.Popen(command('ptb_worker.legacy_conversion'),stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        ready=queue.Queue()
        threading.Thread(target=lambda:ready.put(process.stdout.readline(64)),daemon=True).start()
        pid=int(ready.get(timeout=5))
        handle=win.checked(win.open_process(0x0101,False,pid))
        try:win.checked(win.assign_job(job,handle))
        finally:win.close(handle)
        try:output,_=process.communicate(struct.pack('<I',len(payload))+payload,timeout=timeout)
        except subprocess.TimeoutExpired:raise PreviewError('legacy_conversion_timeout',503) from None
        if process.returncode!=0 or len(output)>MAX_OUTPUT:raise PreviewError('legacy_conversion_budget',413)
        result=json.loads(output)
        if result.get('error') in ('legacy_conversion_budget','legacy_conversion_invalid'):raise PreviewError(result['error'])
        if result.get('schema')!='ptb.lip/1':raise PreviewError('legacy_conversion_invalid')
        return output
    except PreviewError:raise
    except (OSError,ValueError,queue.Empty):raise PreviewError('legacy_conversion_invalid') from None
    finally:
        if job:win.terminate_job(job,1);win.close(job)
        if process:
            if process.poll() is None:process.terminate()
            process.wait(timeout=5);process.stdin.close();process.stdout.close()


def child():
    print(os.getpid(),flush=True)  # parent assigns the Job before releasing input
    from .io.limits import Limits,LimitError
    try:
        size=struct.unpack('<I',sys.stdin.buffer.read(4))[0]
        if not 8<size<=MAX_INPUT:raise LimitError('input_budget')
        raw=sys.stdin.buffer.read(size)
        first,second=struct.unpack_from('<II',raw)
        if not 0<first<=16_000_000 or second>2_000_000 or 8+first+second!=len(raw):raise ValueError('Invalid sizes')
        from .io.lip import convert_local_legacy_lip
        output=convert_local_legacy_lip(raw[8:8+first],Limits(),companion=raw[8+first:] if second else None)
    except (LimitError,MemoryError):output=b'{"error":"legacy_conversion_budget"}'
    except Exception:output=b'{"error":"legacy_conversion_invalid"}'
    if len(output)>MAX_OUTPUT:raise SystemExit(2)
    sys.stdout.buffer.write(output);sys.stdout.buffer.flush()


if __name__=='__main__':child()
