"""Execute trusted core operations outside API/worker memory, with bounded shutdown."""
import json
import logging
import sqlite3
import psycopg
from ptb_worker.process_entry import command
import queue
import subprocess
import sys
import threading
import time
from phonetic_core import __version__ as core_version


def stop_child(child):
    try:
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
    finally:
        for stream in (child.stdin,child.stdout):
            if stream is not None:stream.close()


def execute_claim(store, claim, worker_id, stop, *, step_delay=0):
    snapshot=json.loads(claim['snapshot'])
    if snapshot['operation']=='recording_audio':
        from .recording_jobs import execute
        execute(store,claim,worker_id,stop)
        return
    if snapshot['operation']=='lip_analysis':
        from .m05_executor import execute_claim as execute_m05
        execute_m05(store,claim,worker_id,stop)
        return
    if snapshot['operation']=='mfa_alignment':
        from .m11_executor import execute_claim as execute_m11
        execute_m11(store,claim,worker_id,stop)
        return
    if snapshot['operation']=='phonation_synthesis':
        from .m07_executor import execute_claim as execute_m07
        execute_m07(store,claim,worker_id,stop)
        return
    if snapshot['operation']=='speech_synthesis':
        from .m06_executor import execute_claim as execute_m06
        execute_m06(store,claim,worker_id,stop)
        return
    if snapshot['operation']=='phonology_induction':
        from .m14_executor import execute_claim as execute_m14
        execute_m14(store,claim,worker_id,stop)
        return
    if snapshot['operation'] in ('acoustic_analysis','textgrid_segment','spectrogram_to_audio','egg_analysis','lpc_analysis','pitch_manipulation'):
        from .acoustic_executor import execute_acoustic_claim
        execute_acoustic_claim(store,claim,worker_id,stop)
        return
    if snapshot['operation'] != 'pipeline_check':
        from .file_executor import execute_file_claim
        execute_file_claim(store,claim,worker_id,stop,step_delay=step_delay)
        return
    identity=(claim['id'],worker_id,claim['generation'])
    if snapshot['core_version']!=core_version:
        store.finish(*identity,error='core_version_mismatch');return
    child=subprocess.Popen(command('ptb_worker.core_child'),stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    messages=queue.Queue(maxsize=64)
    def reader():
        try:
            while True:
                line=child.stdout.readline(4097)
                if not line:break
                if len(line)>4096:raise ValueError('Oversized core response')
                messages.put(json.loads(line))
        except (ValueError,OSError):messages.put({'invalid':True})
        finally:messages.put({'eof':True})
    reader_thread=threading.Thread(target=reader,daemon=True)
    try:
        child.stdin.write(json.dumps({'snapshot':snapshot,'step_delay':step_delay})+'\n');child.stdin.flush()
        reader_thread.start()
        progress=0.0;result=None;last_beat=0.0;eof=False
        while True:
            if stop.is_set():
                # Host shutdown is an interruption, not a user's cancellation.
                return
            if time.monotonic()-last_beat>=min(1,store.lease_seconds/3):
                state=store.heartbeat(*identity,progress)
                last_beat=time.monotonic()
                if state!='running':
                    if state=='cancel_requested':store.finish(*identity,error='cancelled')
                    return
            try:message=messages.get(timeout=0.1)
            except queue.Empty:continue
            if message.get('invalid'):store.finish(*identity,error='execution_failed');return
            if 'progress' in message:progress=min(0.99,float(message['progress']))
            if 'result' in message:result=message['result']
            if message.get('eof'):eof=True
            if eof:
                child.wait(timeout=5)
                if child.returncode==0 and result is not None:
                    store.finish(*identity,result=result)
                else:store.finish(*identity,error='execution_failed')
                return
    finally:
        stop_child(child)
        if reader_thread.ident is not None:reader_thread.join(timeout=5)


def run_worker(store, stop, worker_id, *, step_delay=0):
    from ptb_api.quota import StorageError
    idle_delay=0.1
    while not stop.is_set():
        try:claim=store.claim(worker_id)
        except StorageError as exc:
            if exc.code not in ('storage_busy','local_storage_busy'):raise
            # Upload, cleanup and claim share the same store lock. Contention
            # must leave the worker available for the next queue poll.
            stop.wait(0.1)
            continue
        except (sqlite3.OperationalError,psycopg.OperationalError):
            logging.getLogger(__name__).warning('worker_store_retry_required')
            stop.wait(1)
            continue
        if claim is None:
            stop.wait(idle_delay)
            idle_delay=min(1.0,idle_delay*2)
            continue
        idle_delay=0.1
        try:execute_claim(store,claim,worker_id,stop,step_delay=step_delay)
        except Exception:
            # Do not log private snapshots, source paths or exception messages.
            logging.getLogger(__name__).error('worker_execution_failed job=%s',claim['id'])
            files=getattr(store,'files',None)
            if files is not None:
                try:files.fail((claim['id'],worker_id,claim['generation']),'execution_failed')
                except Exception:logging.getLogger(__name__).warning('worker_output_cleanup_retry_required')
            try:store.finish(claim['id'],worker_id,claim['generation'],error='execution_failed')
            except Exception:
                # The lease remains the recovery boundary if storage is unavailable.
                logging.getLogger(__name__).warning('worker_failure_record_retry_required')
                stop.wait(1)
