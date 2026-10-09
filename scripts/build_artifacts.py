"""Mark owned build/staging work from its first write through success or failure."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path


@contextmanager
def build_workspace(path, kind):
    marker = Path(path) / 'artifact-lifecycle.json'
    record = dict(schema='ptb-build-work/1', kind=kind, owner_pid=os.getpid(),
                  state='active', started_at=datetime.now(timezone.utc).isoformat())
    def write():
        marker.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    write()
    try:
        yield
    except BaseException as error:
        record.update(state='failed', error_type=type(error).__name__)
        raise
    else:
        record['state'] = 'completed'
    finally:
        record['finished_at'] = datetime.now(timezone.utc).isoformat()
        write()
