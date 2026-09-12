"""Explicit first-use initialization of a new owned local workspace only.

Authorized 2026-09-12. Existing databases are validated, never migrated here.
"""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
from .io.scratch import no_links
from .local_acoustic_files import initialize_local_files, LocalAcousticFiles
from .store import SQLiteJobStore
from .acoustic_batches import AcousticBatches


def prepare_workspace(root, migrations):
    root = Path(root).absolute()
    no_links(root)
    marker = root / 'workspace.json'
    database, cache = root / 'jobs.sqlite3', root / 'files'
    if not root.exists():
        # Exclusive creation is also a first-run lock: another instance cannot
        # initialize or use the partial tree until workspace.json is published.
        root.mkdir(parents=True, exist_ok=False)
        with database.open('xb'):
            pass
        with closing(sqlite3.connect(database)) as connection:
            connection.execute('PRAGMA foreign_keys=ON')
            for name in ('002_jobs_sqlite.sql', '005_acoustic_batches_sqlite.sql'):
                connection.executescript((Path(migrations) / name).read_text(encoding='utf-8'))
        cache.mkdir()
        initialize_local_files(cache)
        with marker.open('x', encoding='utf-8') as output:
            json.dump({'kind': 'ptb-local-workspace', 'version': 1}, output)
    no_links(marker); no_links(database); no_links(cache)
    try:
        if json.loads(marker.read_text('utf-8')) != {'kind': 'ptb-local-workspace', 'version': 1}:
            raise ValueError()
        if not database.is_file() or not cache.is_dir():
            raise ValueError()
    except (OSError, ValueError):
        raise RuntimeError('本地任务目录尚未初始化完成或不属于本应用；现有文件保持不变。') from None
    jobs = SQLiteJobStore(database)
    jobs.check_schema()
    LocalAcousticFiles(jobs, cache)
    AcousticBatches(jobs, jobs.files)
    return database, cache
