"""Seven-day maintenance restricted to registered updater UUID directories."""
import json
import math
from pathlib import Path
import re
import shutil
import time
from .update_apply import plain_path, read_json, owned_tree
from .updates import _atomic_json

RETENTION_SECONDS = 7 * 24 * 3600
PENDING_STATES = {'awaiting-close', 'waiting', 'launching'}


def clean_update_cache(root, *, protected_downloads=(), now=None, remove=shutil.rmtree, all_cache=False):
    root = Path(root).absolute()
    now = time.time() if now is None else now
    index_path = plain_path(root / 'cache-index.json', root)
    try:
        index = read_json(index_path)
        entries = index.get('entries', {}) if index.get('schema') == 'ptb-update-cache/1' else {}
        if not isinstance(entries, dict): entries = {}
    except (OSError, ValueError):
        entries = {}
    protected = {'downloads/' + name for name in protected_downloads if re.fullmatch('[0-9a-f]{32}', name)}
    folders = []
    failures = []
    # A pending plan owns its helper and package until cancellation or a
    # recorded terminal outcome. Missing/malformed plans err toward retention.
    all_protected = False
    apply_root = plain_path(root / 'apply', root)
    if apply_root.is_dir():
        for folder in apply_root.iterdir():
            if not re.fullmatch('[0-9a-f]{32}', folder.name): continue
            try:
                plain_path(folder, root)
                state = read_json(plain_path(folder / 'status.json', root)).get('state')
                if state in PENDING_STATES or state not in {'failed','cancelled','started'}:
                    protected.add('apply/' + folder.name)
                    plan = read_json(plain_path(folder / 'request.json', root))
                    for key, family in (('helperExe','helpers'), ('package','downloads')):
                        path = plain_path(plan[key], root / family)
                        if len(path.relative_to(root / family).parts) != 2:
                            raise ValueError('layout')
                        protected.add(family + '/' + path.parent.name)
            except Exception:
                all_protected = True
                failures.append({'family':'apply','code':'IDENTITY_UNCONFIRMED'})
    for family in ('downloads','helpers','apply'):
        parent = plain_path(root / family, root)
        if not parent.is_dir(): continue
        for folder in parent.iterdir():
            if not re.fullmatch('[0-9a-f]{32}', folder.name): continue
            key = family + '/' + folder.name
            try:
                plain_path(folder, root)
                if not folder.is_dir(): continue
                identity = [folder.stat().st_dev, folder.stat().st_ino]
                entry = entries.get(key)
                if not isinstance(entry,dict) or entry.get('identity') != identity or type(entry.get('firstSeen')) not in (float,int) or not math.isfinite(entry['firstSeen']) or entry['firstSeen'] < 0:
                    entries[key] = {'identity':identity,'firstSeen':now}
                folders.append((key,folder))
            except Exception:
                failures.append({'family':family,'code':'PATH_UNSAFE'})
    removed, retained = 0, 0
    for key, folder in folders:
        age = now - entries[key]['firstSeen']
        if all_protected or key in protected or (not all_cache and age < RETENTION_SECONDS):
            retained += 1
            continue
        try:
            # Validate the entire final tree before a recursive operation.
            for item in owned_tree(folder, root):
                pass
            remove(folder)
            if folder.exists(): raise OSError('not removed')
            entries.pop(key,None)
            removed += 1
        except Exception:
            retained += 1
            failures.append({'family':key.split('/')[0],'code':'REMOVE_FAILED'})
    _atomic_json(index_path, {'schema':'ptb-update-cache/1','entries':entries})
    summary = {'state':'partial' if failures else 'complete','checkedAt':now,'removedDirectories':removed,
               'retainedDirectories':retained,'failures':failures,'retentionDays':7}
    _atomic_json(root/'cache-status.json',summary)
    return summary
