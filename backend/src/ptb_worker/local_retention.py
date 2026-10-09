"""Opt-in policy for this instance's managed results, never source directories.

The existing asset lock and SQLite transaction fence each sweep against workers.
No schema change is required. Old assets receive a full first-observation grace.
"""
import json
import math
import os
from pathlib import Path
import time
from uuid import UUID
from .io.scratch import no_links
from .store import canonical, JobError

DAY = 86400
ACTIVE = frozenset({'queued', 'running', 'cancel_requested'})
# A captured recording may be the sole original; it is never a result cache.
PROTECTED_OPERATIONS = frozenset({'lip_analysis'})


class LocalRetention:
    def __init__(self, files):
        self.files = files

    def _load(self, state):
        path = self.files.root / '.ptb-retention.json'
        no_links(path)
        if not path.exists():
            return dict(schema='ptb-retention/1', instance_id=state['instance_id'],
                        enabled=True, days=30, exported_cache_days=7, last_sweep=0,
                        records={}, mfa_attempts={}, last_result=None)
        if not path.is_file() or path.stat().st_nlink != 1 or path.stat().st_size > 16_000_000:
            raise JobError('local_retention_metadata_rejected', 503)
        value = json.loads(path.read_text('utf-8'))
        if (value.get('schema') != 'ptb-retention/1' or
                value.get('instance_id') != state['instance_id'] or
                type(value.get('enabled')) is not bool or
                type(value.get('days')) is not int or not 1 <= value['days'] <= 3650 or
                type(value.get('exported_cache_days')) is not int or
                not 1 <= value['exported_cache_days'] <= 3650 or
                not isinstance(value.get('records'), dict)):
            raise JobError('local_retention_metadata_rejected', 503)
        for key, record in value['records'].items():
            try:UUID(key)
            except (ValueError,TypeError):raise JobError('local_retention_metadata_rejected',503) from None
            if not isinstance(record,dict):raise JobError('local_retention_metadata_rejected',503)
            for field in ('first_seen','created_at','touched_at','exported_at'):
                number=record.get(field)
                if field=='exported_at' and number is None:continue
                if type(number) not in (int,float) or not math.isfinite(number) or number<=0:
                    raise JobError('local_retention_metadata_rejected',503)
        from .mfa_retention import ATTEMPT
        attempts=value.setdefault('mfa_attempts',{})
        if not isinstance(attempts,dict):raise JobError('local_retention_metadata_rejected',503)
        for name,record in attempts.items():
            match=ATTEMPT.fullmatch(name)
            if (not match or not isinstance(record,dict) or record.get('job_id')!=match[1] or
                    type(record.get('generation')) is not int or record['generation']!=int(match[2]) or
                    not isinstance(record.get('registry'),str) or not Path(record['registry']).is_absolute() or
                    type(record.get('size_bytes')) is not int or record['size_bytes']<0 or
                    type(record.get('removed')) is not bool or not isinstance(record.get('directory_id'),list) or
                    len(record['directory_id'])!=2 or any(type(v) is not int or v<0 for v in record['directory_id']) or record['directory_id'][1]<=0):
                raise JobError('local_retention_metadata_rejected',503)
            for field in ('first_seen','created_at','completed_at','removed_at'):
                number=record.get(field)
                if field in ('completed_at','removed_at') and number is None:continue
                if type(number) not in (int,float) or not math.isfinite(number) or number<=0:
                    raise JobError('local_retention_metadata_rejected',503)
        return value

    def _save(self, value):
        root = self.files.root
        pending, target = root / '.ptb-retention.next', root / '.ptb-retention.json'
        no_links(pending); no_links(target)
        raw = canonical(value).encode('utf-8')
        if len(raw) > 16_000_000:
            raise JobError('local_retention_metadata_rejected', 503)
        if pending.exists() and (not pending.is_file() or pending.stat().st_nlink != 1):
            raise JobError('local_retention_metadata_rejected', 503)
        with pending.open('wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(pending, target)

    @staticmethod
    def _time(value, fallback):
        return value if type(value) in (int, float) and math.isfinite(value) and 0 < value <= fallback else fallback

    def _observe(self, state, policy, now):
        for key, asset in state['assets'].items():
            if asset['kind'] != 'result' or asset['state'] != 'ready':
                continue
            if key not in policy['records']:
                # New releases stamp creation. A historic mtime is not evidence.
                created = self._time(asset.get('retention_created_at'), now)
                policy['records'][key] = dict(first_seen=now, created_at=created,
                                             exported_at=None, touched_at=created)

    def status(self, *, now=None):
        now = time.time() if now is None else now
        with self.files.batch_transaction() as tx:
            state = self.files.state; value = self._load(state)
            self._observe(state, value, now); self._save(value)
            assets = [a for a in state['assets'].values() if a['state'] == 'ready']
            return dict(schema=value['schema'], enabled=value['enabled'], days=value['days'],
                        exported_cache_days=value['exported_cache_days'],
                        result_bytes=sum(a['size_bytes'] for a in assets if a['kind'] == 'result'),
                        input_bytes=sum(a['size_bytes'] for a in assets if a['kind'] == 'input'),
                        result_count=sum(a['kind'] == 'result' for a in assets),
                        diagnostic_days=7,
                        diagnostic_count=sum(not r['removed'] for r in value.get('mfa_attempts',{}).values()),
                        diagnostic_bytes=sum(r['size_bytes'] for r in value.get('mfa_attempts',{}).values() if not r['removed']),
                        last_result=value.get('last_result'))

    def register_mfa_attempt(self,path,identity,*,now=None):
        now=time.time() if now is None else now
        if type(now) not in (int,float) or not math.isfinite(now) or now<=0:raise JobError('mfa_diagnostic_time_rejected',409)
        from .mfa_retention import register
        with self.files.batch_transaction() as tx:
            job=self.files._fence(tx,identity)
            snapshot=json.loads(job['snapshot'])
            if snapshot.get('operation')!='mfa_alignment' or snapshot.get('execution_route')!='desktop-local':
                raise JobError('mfa_diagnostic_owner_rejected',409)
            state=self.files.state;value=self._load(state)
            register(path,identity,state,value,now);self._save(value)

    def complete_mfa_attempt(self,path,identity,*,now=None):
        now=time.time() if now is None else now
        if type(now) not in (int,float) or not math.isfinite(now) or now<=0:raise JobError('mfa_diagnostic_time_rejected',409)
        from .mfa_retention import completed
        with self.files.batch_transaction():
            state=self.files.state;value=self._load(state)
            completed(path,identity,state,value,now);self._save(value)

    def configure(self, *, enabled, days, exported_cache_days=7):
        if (type(enabled) is not bool or type(days) is not int or not 1 <= days <= 3650 or
                type(exported_cache_days) is not int or not 1 <= exported_cache_days <= 3650):
            raise JobError('invalid_retention_policy', 422)
        with self.files.locked() as state:
            value = self._load(state)
            value.update(enabled=enabled, days=days, exported_cache_days=exported_cache_days)
            self._save(value)
        return self.status()

    def exported(self, asset_ids, *, now=None):
        now = time.time() if now is None else now
        keys = [str(UUID(key)) for key in asset_ids]
        if not keys or len(keys) > 3004 or len(keys) != len(set(keys)):
            raise JobError('invalid_export_receipt', 422)
        with self.files.batch_transaction() as tx:
            state = self.files.state; value = self._load(state)
            self._observe(state, value, now)
            for key in keys:
                asset = self.files._asset(key)
                if asset['kind'] != 'result':
                    raise JobError('invalid_export_receipt', 422)
                self.files._readable(tx, asset)
            for key in keys:
                value['records'][key]['exported_at'] = now
            self._save(value)
        return dict(acknowledged=len(keys))

    @staticmethod
    def _references(value, candidates):
        found, pending = set(), [value]
        while pending:
            item = pending.pop()
            if isinstance(item, dict): pending.extend(item.values())
            elif isinstance(item, list): pending.extend(item)
            elif isinstance(item, str) and item in candidates: found.add(item)
        return found

    def sweep(self, *, now=None, force=False, all_cache=False):
        now = time.time() if now is None else now
        files = self.files
        with files.batch_transaction() as tx:
            state = files.state; policy = self._load(state)
            self._observe(state, policy, now)
            if not all_cache and (not policy['enabled'] or (not force and now - policy['last_sweep'] < DAY)):
                self._save(policy)
                return dict(skipped=True, count=0, bytes=0, enabled=policy['enabled'],diagnostic_count=0,diagnostic_bytes=0,diagnostic_failed_count=0,diagnostic_days=7)
            jobs = {row['id']: dict(row) for row in tx.execute('SELECT * FROM {jobs}').fetchall()}
            if all_cache and any(row['state'] in ACTIVE for row in jobs.values()):
                raise JobError('local_cache_tasks_active', 409)
            assets = {key: asset for key, asset in state['assets'].items()
                      if asset['kind'] == 'result' and asset['state'] == 'ready'}
            eligible = set()
            for key, asset in assets.items():
                job = jobs.get(asset['job_id']); record = policy['records'][key]
                if not job or job['state'] != 'succeeded' or job['generation'] != asset['generation']:
                    continue
                snapshot = json.loads(job['snapshot'])
                if snapshot.get('operation') in PROTECTED_OPERATIONS:
                    continue
                touched = max(record['created_at'], record['touched_at'],
                              self._time(asset.get('retention_touched_at', record['created_at']), now))
                exported = record.get('exported_at')
                age = policy['exported_cache_days'] if exported else policy['days']
                if all_cache or now - max(touched, exported or 0) >= age * DAY:
                    eligible.add(key)
            # Keep a task's companion files together. Exporting one WAV cannot
            # silently discard the metadata and the other unsaved outputs.
            bundles = {}
            for key, asset in assets.items():
                bundles.setdefault(asset['job_id'], set()).add(key)
            for bundle in bundles.values():
                if not bundle <= eligible: eligible.difference_update(bundle)
            # Surviving results and active consumers protect their complete input
            # chain, including M07 analysis_job_id and other parent job IDs.
            protected_jobs = {row['id'] for row in jobs.values() if row['state'] in ACTIVE}
            protected_jobs.update(a['job_id'] for key, a in assets.items() if key not in eligible)
            protected, visited, queue = set(), set(), list(protected_jobs)
            while queue:
                key = queue.pop()
                if key in visited or key not in jobs: continue
                visited.add(key); snapshot = json.loads(jobs[key]['snapshot'])
                protected.update(self._references(snapshot, set(assets)))
                parents = self._references(snapshot, set(jobs))
                for parent in parents:
                    protected.update(k for k, a in assets.items() if a['job_id'] == parent)
                queue.extend(parents - visited)
            selected=sorted(eligible-protected)
            # Validate the complete set before any unlink. A bad later path or
            # an externally changed result must not cause an unreported prefix.
            for key in selected:
                asset = assets[key]
                # _path checks UUID, resolved ownership, every ancestor, links and
                # link count. The asset lock stays held until state is committed.
                path = files._path(key)
                if path.exists() and path.stat().st_size != asset['size_bytes']:
                    raise JobError('local_result_changed', 409)
            removed, freed, failed = [], 0, []
            for key in selected:
                asset=assets[key];size=asset['size_bytes']
                try:files._delete(asset)
                except OSError:
                    # An access-denied file remains retryable, with its identity
                    # and original expiry policy. Report partial success.
                    if files._path(key).exists():
                        asset['state']='ready';files._save(state)
                    failed.append(key);continue
                removed.append(key); freed += size
            from .mfa_retention import sweep as sweep_diagnostics
            diagnostics=sweep_diagnostics(state,policy,jobs,now)
            result = dict(skipped=False, count=len(removed)+diagnostics['diagnostic_count'], bytes=freed+diagnostics['diagnostic_bytes'],
                          protected_count=len(eligible & protected), failed_count=len(failed)+diagnostics['diagnostic_failed_count'],
                          complete=not failed and not diagnostics['diagnostic_failed_count'],at=now,**diagnostics)
            policy.update(last_sweep=now, last_result=result); self._save(policy)
            return result
