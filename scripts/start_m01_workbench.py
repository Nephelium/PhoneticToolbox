"""Development workbench using the reviewed existing test DB; never runs DDL."""
import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    from ptb_worker.store import SQLiteJobStore
    from ptb_worker.acoustic_batches import AcousticBatches
    from ptb_worker.local_acoustic_files import LocalAcousticFiles,initialize_local_files
    from ptb_worker.io.scratch import no_links
    base=ROOT/'output/validation/m01';no_links(base)
    jobs=SQLiteJobStore(ROOT/'output/validation/p06/local-state.sqlite3');jobs.check_schema()
    with jobs.transaction(write=False) as tx:
        assert [dict(r) for r in tx.execute('SELECT version FROM acoustic_batch_version')]==[{'version':1}],'005 schema is not initialized'
    config=base/'workbench-local.json';no_links(config)
    if not config.exists():
        cache=base/('workbench-cache-'+uuid4().hex);cache.mkdir();initialize_local_files(cache)
        with config.open('x',encoding='utf-8') as f:
            json.dump({'version':1,'cache':cache.name},f);f.write('\n');f.flush();os.fsync(f.fileno())
    value=json.loads(config.read_text('utf-8'));name=value['cache']
    if value['version']!=1 or not isinstance(name,str) or not name.startswith('workbench-cache-') or Path(name).name!=name:raise ValueError('Invalid local cache config')
    cache=base/name;no_links(cache)
    reaper=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe'
    files=LocalAcousticFiles(jobs,cache,reaper_binary=reaper);AcousticBatches(jobs,files)
    if args.prepare_only:print(json.dumps({'ready':True,'cache':str(cache)}));return
    from ptb_desktop.host import run
    raise SystemExit(run(ROOT/'frontend/dist',jobs_path=jobs.path,local_files_root=cache,reaper_binary=reaper))


if __name__=='__main__':main()
