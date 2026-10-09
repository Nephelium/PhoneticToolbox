"""M09 bounded, memory-only preview through owned Windows process handles."""
import json
import os
import queue
import subprocess
import sys
import threading
from .spectrogram_preview import PreviewError
from .process_entry import command


def render(payload, config, timeout=40):
    if not 0 < len(payload) <= 64_000_000:raise PreviewError('spectrogram_budget')
    if os.name != 'nt':raise PreviewError('preview_platform_unverified',503)
    from .native import windows as win
    process=None;job=None
    try:
        job=win.checked(win.create_job(None,None));limits=win.ExtendedLimit()
        limits.basic.flags=0x2000|0x100|0x200
        limits.process_memory=1_000_000_000;limits.job_memory=1_000_000_000
        win.checked(win.set_job(job,9,win.c.byref(limits),win.c.sizeof(limits)))
        process=subprocess.Popen(command('ptb_worker.spec2wav_preview'),stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW,
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                 'OPENCV_IO_MAX_IMAGE_PIXELS':'25000000'})
        ready=queue.Queue()
        threading.Thread(target=lambda:ready.put(process.stdout.readline(64)),daemon=True).start()
        pid=int(ready.get(timeout=5));handle=win.checked(win.open_process(0x0101,False,pid))
        try:win.checked(win.assign_job(job,handle))
        finally:win.close(handle)
        header=json.dumps(dict(size=len(payload),config=config)).encode()+b'\n'
        if len(header)>1_000_000:raise PreviewError('spectrogram_budget')
        try:output,_=process.communicate(header+payload,timeout=timeout)
        except subprocess.TimeoutExpired:raise PreviewError('preview_timeout',503) from None
        if process.returncode!=0 or len(output)>12_000_000:raise PreviewError('preview_resource_failed',503)
        result=json.loads(output)
        if 'error' in result:raise PreviewError(result['error'])
        return result
    except PreviewError:raise
    except (OSError,ValueError,queue.Empty):raise PreviewError('preview_resource_failed',503) from None
    finally:
        if job:win.terminate_job(job,1);win.close(job)
        if process:
            if process.poll() is None:process.terminate()
            process.wait(timeout=5)
            process.stdin.close();process.stdout.close()


def preview_source(store,owner,body):
    from .store import JobError
    from .segmentation import digest
    from .spectrogram_preview import preview_slot
    role='audio' if body.config.mode=='audio_draw' else 'image'
    if store.files is None:raise JobError('task_service_unavailable',503)
    with preview_slot():
        with store.files.batch_transaction() as tx:
            assets=store.files.validate_batch_input(tx,owner,body.project_id,{role:body.image.model_dump()},tx.now())
        source=assets[0]
        if source['size_bytes']>(64_000_000 if role=='audio' else 16_000_000):raise JobError('spectrogram_budget',422)
        if store.postgres:
            read=lambda offset,size:store.files.storage.read_block(owner,source['id'],offset,size)
        else:read=lambda offset,size:store.files.read_result(owner,source['id'],offset,size)
        from ptb_api.quota import CHUNK_BYTES
        raw=b''.join(read(offset,min(CHUNK_BYTES,source['size_bytes']-offset)) for offset in range(0,source['size_bytes'],CHUNK_BYTES))
        if len(raw)!=source['size_bytes'] or digest(raw)!=body.image.sha256:raise JobError('input_unavailable',409)
        result=render(raw,body.config.model_dump())
        # Recheck ownership/hash/expiry after potentially slow native computation.
        with store.files.batch_transaction() as tx:
            store.files.validate_batch_input(tx,owner,body.project_id,{role:body.image.model_dump()},tx.now())
        return result


def main():
    print(os.getpid(),flush=True)
    try:
        header=json.loads(sys.stdin.buffer.readline(1_000_000));size=header['size']
        if type(size)!=int or not 0<size<=64_000_000:raise ValueError('spectrogram_budget')
        payload=sys.stdin.buffer.read(size)
        if len(payload)!=size:raise ValueError('input_unavailable')
        from .spec2wav_child import preview
        result=preview(payload,header['config'])
    except Exception as exc:
        from .acoustic_errors import public_error
        result={'error':public_error(exc)}
    output=json.dumps(result,allow_nan=False).encode()
    if len(output)>12_000_000:raise SystemExit(2)
    sys.stdout.buffer.write(output);sys.stdout.buffer.flush()


if __name__=='__main__':main()
