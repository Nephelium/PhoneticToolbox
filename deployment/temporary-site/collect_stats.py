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
from bisect import bisect_right
from collections import Counter


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
    if row.get('kind')=='web':
        return 'web' if row.get('method')=='POST' and row.get('status')==200 and row.get('webUse')=='1' else None
    if row.get('method') != 'GET' or row.get('kind') not in ('page', 'portable', 'installer', 'macos'):
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


def backfill_macos(state, log_dir):
    """Once, recover only provable full GETs; combined logs lack Range headers.

    Raw access logs stay on the server. Persist only timestamps and aggregates.
    Already anonymously logged requests are subtracted before importing counts.
    """
    if 'macosBackfill' in state:
        return
    anonymous = Counter()
    for path in sorted(log_dir.glob('ptb-website-stats.log*')):
        if not path.is_file() or path.is_symlink():
            continue
        with (gzip.open if path.suffix == '.gz' else open)(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    if kind_of(row) == 'macos':
                        anonymous[(int(datetime.fromisoformat(row['time']).timestamp()), row['status'], row['bytes'])] += 1
                except (ValueError, KeyError, TypeError):
                    continue
    remaining = anonymous.copy()
    events, seen, skipped_ranges = [], set(), 0
    pattern = re.compile(r'\[([^\]]+)\] "([A-Z]+) ([^ ]+) [^\"]+" (\d+) (\d+|-) "[^\"]*" "([^\"]*)"')
    mac_path = re.compile(r'^/releases/macos-arm64/[^/]+/PhoneticToolbox-[^/]+-macos[^/]+\.zip$')
    for path in sorted(log_dir.glob('access.log*')):
        if not path.is_file() or path.is_symlink():
            continue
        with (gzip.open if path.suffix == '.gz' else open)(path, 'rt', encoding='utf-8', errors='replace') as stream:
            first = stream.readline()
            key = hashlib.sha256(first.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            stream.seek(0)
            for line in stream:
                match = pattern.search(line)
                if not match:
                    continue
                time, method, url, status, size, agent = match.groups()
                if method != 'GET' or not mac_path.fullmatch(url.split('?', 1)[0]):
                    continue
                if re.search(r'bot|crawler|spider|slurp|headless|PTB-Website-Verify', agent, re.I):
                    continue
                if status == '206':
                    skipped_ranges += 1
                    continue
                if status != '200' or size == '-' or int(size) <= 0:
                    continue
                stamp = int(datetime.strptime(time, '%d/%b/%Y:%H:%M:%S %z').timestamp())
                signature = (stamp, 200, int(size))
                if remaining[signature]:
                    remaining[signature] -= 1
                else:
                    events.append(stamp)
    all_events = sorted(events + [stamp for (stamp, _, _), count in anonymous.items() for _ in range(count)])
    tracked = all_events[0] if all_events else int(datetime.fromisoformat(now()).timestamp())
    history = {p[0]: [*p[:5], None] for p in state.get('history', [])}
    for stamp in sorted(set(all_events)):
        history.setdefault(stamp, [stamp, None, None, None, None, None])
    for stamp, point in history.items():
        point[5] = bisect_right(all_events, stamp) if stamp >= tracked else None
    state['history'] = sorted(history.values(), key=lambda p: p[0])
    state['counts']['macos'] += len(events)
    state['macosStartedAt'] = datetime.fromtimestamp(tracked, timezone.utc).isoformat(timespec='seconds')
    state['macosBackfill'] = {'imported': len(events), 'skippedUnverifiableRanges': skipped_ranges,
                            'completedAt': now(), 'source': 'retained-combined-log-GET-200'}


def history_points(log_dir,counts):
    """Recover only retained, timestamped anonymous log events; no invented past."""
    events={}
    for path in sorted(log_dir.glob('ptb-website-stats.log*')):
        if not path.is_file() or path.is_symlink():continue
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rb') as stream:
            for line in stream:
                try:
                    row=json.loads(line);kind=kind_of(row)
                    stamp=int(datetime.fromisoformat(row['time']).timestamp())
                    if kind and kind != 'macos':events[(stamp,row['id'])]=(stamp,kind)
                except (ValueError,TypeError,KeyError):continue
    totals={k:0 for k in ('page','portable','installer','web')}
    for _,kind in events.values():totals[kind]+=1
    if any(totals[k]>counts.get(k,0) for k in totals):return []
    running={k:counts.get(k,0)-totals[k] for k in totals};points={};web_started=False
    for stamp,kind in sorted(events.values()):
        running[kind]+=1;web_started=web_started or kind=='web'
        # Keep the last actual observation within each minute, not a guessed value.
        points[stamp//60]=[stamp,running['page'],running['portable']+running['installer'],running['web'] if web_started else None,None]
    return list(points.values())


def compact_history(points,current):
    columns = max((len(point) for point in points), default=5)
    points = [point + [None] * (columns - len(point)) for point in points]
    buckets={}
    for point in sorted(points,key=lambda p:p[0]):
        age=current-point[0]
        width=60 if age<2*86400 else 3600 if age<90*86400 else 86400
        bucket=buckets.setdefault((width,point[0]//width),{})
        # Each metric has its own sampling cadence. Keep its last real sample,
        # at the original timestamp, even when a later row has no Star sample.
        for column in range(1,columns):
            if point[column] is not None:bucket[column]=point
    retained={}
    for bucket in buckets.values():
        for point in bucket.values():
            previous=retained.get(point[0],[point[0]]+[None]*(columns-1))
            retained[point[0]]=[point[0],*[point[i] if point[i] is not None else previous[i] for i in range(1,columns)]]
    return sorted(retained.values(),key=lambda p:p[0])


def github_stars():
    from urllib.request import Request,urlopen
    request=Request('https://api.github.com/repos/Nephelium/PhoneticToolbox',headers={'Accept':'application/vnd.github+json','User-Agent':'PhoneticToolbox-Website-Stats'})
    with urlopen(request,timeout=8) as response:
        data=json.loads(response.read(131072))
    count=data['stargazers_count']
    if type(count) is not int or count<0:raise ValueError('Invalid GitHub statistics')
    return count


def collect(log_dir, state_path, public_path, *, fetch_stars=None, backfill_access_logs=False):
    if state_path.exists():
        state = json.loads(state_path.read_text('utf-8'))
        assert state['schema'] == 1
    else:
        state = {'schema': 1, 'startedAt': now(), 'counts': {'page': 0, 'portable': 0, 'installer': 0}, 'offsets': {}, 'invalidLines': 0}
    state['counts'].setdefault('web',0)
    state['counts'].setdefault('macos',0)
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
    counts = state['counts']
    stamp=int(datetime.fromisoformat(state['updatedAt']).timestamp())
    if 'history' not in state:state['history']=history_points(log_dir,counts)
    if backfill_access_logs:
        backfill_macos(state, log_dir)
    stars=state.get('github',{})
    last=int(datetime.fromisoformat(stars['updatedAt']).timestamp()) if stars.get('updatedAt') else 0
    if fetch_stars is not None and stamp-state.get('githubAttempt',0)>=300:
        state['githubAttempt']=stamp
        try:state['github']={'count':fetch_stars(),'updatedAt':state['updatedAt']};stars=state['github'];last=stamp
        except Exception:pass  # Preserve the last sample/time; do not fabricate zero.
    macos_enabled = bool(state.get('macosStartedAt')) or counts['macos'] > 0
    if macos_enabled:
        state.setdefault('macosStartedAt', state['updatedAt'])
    point=[stamp,counts['page'],counts['portable']+counts['installer'],counts['web'],stars.get('count') if last==stamp else None,
           counts['macos'] if macos_enabled else None]
    state['history']=compact_history([*state['history'],point],stamp)
    atomic_json(state_path, state, 0o600)
    public = {'schema': 1, 'startedAt': state['startedAt'], 'updatedAt': state['updatedAt'],
              'pageViews': counts['page'], 'downloads': {'total': counts['portable'] + counts['installer'] + counts['macos'],
              'windows': counts['portable'] + counts['installer'], 'macos': counts['macos'] if macos_enabled else None,
              'portable': counts['portable'], 'installer': counts['installer'], 'macosStartedAt': state.get('macosStartedAt')},'webUses':counts['web'],
              'github':stars or None,'historyUpdatedAt':state['updatedAt']}
    atomic_json(public_path, public, 0o644)
    atomic_json(public_path.with_name('site-stats-history.json'),{'schema':1,'columns':['timestamp','pageViews','windowsDownloads','webUses','githubStars','macosDownloads'],
        'updatedAt':state['updatedAt'],'points':state['history'],'source':'anonymous-server-log-and-snapshots'},0o644)
    return public


def main():
    import fcntl  # Linux service; the pure collector can also be tested on Windows.
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=Path, default=Path('/var/log/nginx'))
    parser.add_argument('--state', type=Path, default=Path('/var/lib/phonetictoolbox-website-stats/state.json'))
    parser.add_argument('--public', type=Path, default=Path('/var/www/phonetictoolbox-coming-soon/site-stats.json'))
    parser.add_argument('--backfill-macos-access-logs', action='store_true')
    args = parser.parse_args()
    args.state.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        collect(args.log_dir, args.state, args.public,fetch_stars=github_stars,
                backfill_access_logs=args.backfill_macos_access_logs)


if __name__ == '__main__':
    main()
