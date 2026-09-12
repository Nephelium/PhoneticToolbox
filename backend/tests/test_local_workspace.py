import hashlib
from pathlib import Path
import pytest
from ptb_worker.local_workspace import prepare_workspace

MIGRATIONS = Path(__file__).resolve().parents[1] / 'migrations'


def test_new_dedicated_workspace_and_repeat_no_ddl(tmp_path):
    root = tmp_path / 'owned-new'
    database, cache = prepare_workspace(root, MIGRATIONS)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    assert prepare_workspace(root, tmp_path / 'nonexistent-migrations') == (database, cache)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_existing_unowned_directory_is_never_initialized(tmp_path):
    data = tmp_path / 'user.sqlite'
    data.write_bytes(b'original user data')
    with pytest.raises(RuntimeError):
        prepare_workspace(tmp_path, MIGRATIONS)
    assert data.read_bytes() == b'original user data'
    assert not (tmp_path / 'jobs.sqlite3').exists()
