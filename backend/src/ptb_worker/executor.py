"""Execute trusted core operations outside API/worker memory, with bounded shutdown."""
import json
from ptb_worker.process_entry import command
import queue
import subprocess
import sys
import threading
import time
from phonetic_core import __version__ as core_version


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        child.wait(timeout=5)
    child.stdin.close()
    child.stdout.close()


def execute_claim(store, claim, worker_id, stop, *, step_delay=0):
    snapshot=json.loads(claim['snapshot'])
    if snapshot['operation'] in ('acoustic_analysis','textgrid_segment','spectrogram_to_audio','egg_analysis'):
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
                store.cancel(str(claim['owner_id']),claim['id'])
            if stop.is_set() or time.monotonic()-last_beat>=min(1,store.lease_seconds/3):
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
    while not stop.is_set():
        claim=store.claim(worker_id)
        if claim is None:stop.wait(0.1);continue
        execute_claim(store,claim,worker_id,stop,step_delay=step_delay)
