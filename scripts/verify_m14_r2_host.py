"""Owned M14 UI test transport, using a disposable copy of the existing schema."""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for part in ('scripts', 'tests/support', 'desktop/src', 'backend/src', 'packages/phonetic_core/src'):
    sys.path.insert(0, str(ROOT / part))
from verification_artifacts import verification_run
from ptb_worker.local_acoustic_files import initialize_local_files
import m14_host_bridge

with verification_run('m14', 'r2', 'Synthetic TSV large tables; existing schema copied read-only; production adapter/worker') as (out, scratch):
    db = scratch / 'jobs.sqlite3'
    template = ROOT / 'output/validation/p06/local-state.sqlite3'
    with sqlite3.connect(template.as_uri() + '?mode=ro', uri=True) as source, sqlite3.connect(db) as target:
        source.backup(target)
    cache = scratch / 'cache'
    cache.mkdir()
    initialize_local_files(cache)
    m14_host_bridge.setup = lambda: (scratch, db, cache)
    m14_host_bridge.main()
