"""P06 worker: one private config line; owner EOF requests bounded shutdown."""
import argparse
import json
import sys
import threading
from uuid import uuid4
from .executor import run_worker
from .store import PostgresJobStore, SQLiteJobStore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-step-delay',type=float,default=0)
    args=parser.parse_args()
    if not 0<=args.probe_step_delay<=0.2:parser.error('Probe delay must be 0–0.2 seconds')
    config=json.loads(sys.stdin.readline())
    if config.get('kind') not in ('postgres','sqlite'):raise ValueError('Unsupported task store')
    options={'max_running':config.get('max_running',2),'lease_seconds':config.get('lease_seconds',10)}
    store=PostgresJobStore(config['dsn'],**options) if config['kind']=='postgres' else SQLiteJobStore(config['path'],**options)
    store.check_schema()
    if config.get('storage_root'):
        from ptb_api.storage import Storage
        from .files import FilePipeline
        storage=Storage(config['dsn'],config['storage_root'])
        if config.get('enable_acoustic_batches'):
            from .acoustic_files import AcousticFiles
            from .acoustic_batches import AcousticBatches
            files=AcousticFiles(store,storage);files.reaper_binary=config.get('reaper_binary')
            AcousticBatches(store,files)
        else:FilePipeline(store,storage)
        storage.recover()
    if config.get('local_files_root'):
        from .local_acoustic_files import LocalAcousticFiles
        from .acoustic_batches import AcousticBatches
        files=LocalAcousticFiles(store,config['local_files_root'],reaper_binary=config.get('reaper_binary'))
        AcousticBatches(store,files);files.recover()
    stop=threading.Event()
    cleaner=None
    if config.get('local_files_root'):
        from .local_retention import LocalRetention
        def retention_loop():
            while not stop.is_set():
                try:LocalRetention(files).sweep()
                except Exception:
                    # Keep every uncertain file. A failed sweep is retried and
                    # must never interrupt a research task or expose source paths.
                    import logging
                    logging.getLogger(__name__).warning('local_retention_retry_required')
                stop.wait(3600)
        cleaner=threading.Thread(target=retention_loop,daemon=True)
        cleaner.start()
    def owner_closed():sys.stdin.readline();stop.set()
    threading.Thread(target=owner_closed,daemon=True).start()
    print(json.dumps({'ready':True}),flush=True)
    try:run_worker(store,stop,str(uuid4()),step_delay=args.probe_step_delay)
    finally:
        stop.set()
        if cleaner:cleaner.join(timeout=6)


if __name__=='__main__':
    try:main()
    except Exception:
        print('Worker stopped: storage or execution unavailable; active work will become interrupted after its lease.',file=sys.stderr)
        raise SystemExit(1) from None
