"""Aggregate the site's anonymous nginx log. No visitor identifiers are stored.

Offsets are keyed by the first complete log line, so log rotation and gzip
compression do not count requests twice. A partial last line is retried later.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import os
import re


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def atomic_json(path, value, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def kind_of(row):
    if row.get('method') != 'GET' or row.get('kind') not in ('page', 'portable', 'installer'):
        return None
    if row.get('status') not in (200, 206) or row.get('bytes', 0) <= 0:
        return None
    if row['kind'] == 'page':
        return 'page' if row['status'] == 200 else None
    # Only a whole-file request or a first range initiates a download.
    # Subsequent range chunks/resumes are deliberately excluded.
    if row.get('range', '') not in ('', '-') and not re.fullmatch(r'bytes=0-\d*', row['range']):
        return None
    return row['kind']


def collect(log_dir, state_path, public_path):
    if state_path.exists():
        state = json.loads(state_path.read_text('utf-8'))
        assert state['schema'] == 1
    else:
        state = {'schema': 1, 'startedAt': now(), 'counts': {'page': 0, 'portable': 0, 'installer': 0}, 'offsets': {}, 'invalidLines': 0}
    for path in sorted(log_dir.glob('ptb-website-stats.log*')):
        if not path.is_file() or path.is_symlink():
            continue
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rb') as stream:
            first = stream.readline()
            if not first.endswith(b'\n'):
                continue
            key = hashlib.sha256(first).hexdigest()
            offset = state['offsets'].get(key, 0)
            stream.seek(offset)
            while True:
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    break
                try:
                    row = json.loads(line)
                    kind = kind_of(row)
                    if kind:
                        state['counts'][kind] += 1
                except (ValueError, TypeError, AttributeError):
                    state['invalidLines'] += 1
                offset = stream.tell()
            state['offsets'][key] = offset
    state['updatedAt'] = now()
    atomic_json(state_path, state, 0o600)
    counts = state['counts']
    public = {'schema': 1, 'startedAt': state['startedAt'], 'updatedAt': state['updatedAt'],
              'pageViews': counts['page'], 'downloads': {'total': counts['portable'] + counts['installer'],
              'portable': counts['portable'], 'installer': counts['installer']}}
    atomic_json(public_path, public, 0o644)
    return public


def main():
    import fcntl  # Linux service; the pure collector can also be tested on Windows.
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=Path, default=Path('/var/log/nginx'))
    parser.add_argument('--state', type=Path, default=Path('/var/lib/phonetictoolbox-website-stats/state.json'))
    parser.add_argument('--public', type=Path, default=Path('/var/www/phonetictoolbox-coming-soon/site-stats.json'))
    args = parser.parse_args()
    args.state.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        collect(args.log_dir, args.state, args.public)


if __name__ == '__main__':
    main()
