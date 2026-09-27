"""Opt-in remote/1 coordinator on the existing JobStore transaction authority.

Never mounted by importing this module; legacy claim/recover must be integrated
before enabling it. No DDL, DSN disclosure, science execution or disk paths here.
"""
from dataclasses import dataclass
import hashlib
import json
import secrets
from uuid import uuid4

from .store import JobError, canonical

OPERATIONS = frozenset({'lpc_analysis', 'egg_analysis', 'acoustic_analysis'})
NETWORK_ERRORS = frozenset({'network_error', 'transfer_stalled', 'lease_expired'})
RETRYABLE_ERRORS = NETWORK_ERRORS | {'node_shutdown'}
CHUNK_BYTES = 1_048_576


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class RemotePolicy:
    heartbeat_seconds: int = 10
    lease_seconds: int = 60
    recovery_checks: int = 3
    cooldown_seconds: int = 30
    node_wait_seconds: int = 30
    transfer_seconds: int = 30
    total_active: int = 32
    account_running: int = 1
    server_memory_bytes: int = 1_073_741_824

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in self.__dict__.values()):
            raise ValueError('positive integer policy required')
        if self.lease_seconds < 2 * self.heartbeat_seconds:
            raise ValueError('lease must cover at least two heartbeats')


class RemoteCoordinator:
    def __init__(self, store, files, *, policy=None, server_capability=None, transaction=None):
        self.store, self.files = store, files
        self.policy = policy or RemotePolicy()
        # Return the ACTUAL verified server runtime hash, not a client assertion/bool.
        self.server_capability = server_capability or (lambda operation, runtime: None)
        # A's adapter must provide the combined storage/scheduling transaction.
        self.transaction = transaction or store.transaction

    @staticmethod
    def sql(tx, query, params=()):
        return tx.execute(query.replace('@', 'ptb_jobs.' if tx.postgres else ''), params)

    def register(self, *, scopes, operations, runtime_hash, slots, memory_bytes):
        """SERVER ADMIN ONLY. runtime_hash must bind reviewed runtime evidence."""
        import re
        if (not scopes or any(len(s) != 2 or not all(isinstance(x, str) and x for x in s) for s in scopes)
                or not operations or not set(operations) <= OPERATIONS
                or not re.fullmatch('[0-9a-f]{64}', runtime_hash)
                or type(slots) is not int or not 1 <= slots <= 16
                or type(memory_bytes) is not int or memory_bytes <= 0):
            raise ValueError('invalid administrative admission')
        node, token = str(uuid4()), secrets.token_urlsafe(32)
        with self.transaction() as tx:
            self.sql(tx, '''INSERT INTO @remote_nodes
                (id,token_hash,token_until,scopes,operations,runtime_hash,slots,memory_bytes)
                VALUES(?,?,?,?,?,?,?,?)''', (node, hashlib.sha256(token.encode()).hexdigest(),
                tx.now()+900, canonical(scopes), canonical(operations), runtime_hash, slots, memory_bytes))
        return node, token

    def issue(self, node_id):
        """Explicit administrator rotation; no node self-renewal privilege."""
        token = secrets.token_urlsafe(32)
        with self.transaction() as tx:
            node = self.sql(tx, 'SELECT * FROM @remote_nodes WHERE id=?', (node_id,)).fetchone()
            if not node or node['revoked']: raise JobError('node_unauthorized', 401)
            self.sql(tx, 'UPDATE @remote_nodes SET token_hash=?,token_until=? WHERE id=?',
                     (hashlib.sha256(token.encode()).hexdigest(), tx.now()+900, node_id))
        return token

    def auth(self, tx, token):
        if not isinstance(token, str) or not 32 <= len(token) <= 128:
            raise JobError('node_unauthorized', 401)
        node = self.sql(tx, 'SELECT * FROM @remote_nodes WHERE token_hash=?',
                        (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        if not node or node['revoked'] or node['token_until'] <= tx.now():
            raise JobError('node_unauthorized', 401)
        return dict(node)

    @staticmethod
    def scope(node, job):
        if [str(job['owner_id']), str(job['project_id'])] not in json.loads(node['scopes']):
            raise JobError('node_scope_denied', 403)

    def health(self, token, runtime_hash):
        with self.transaction() as tx:
            node = self.auth(tx, token); now = tx.now()
            if node['runtime_hash'] != runtime_hash: raise JobError('runtime_mismatch', 409)
            gap = now-node['last_seen']
            count = node['healthy_count']
            cool = node['cool_until']
            if gap > self.policy.heartbeat_seconds*3:
                count = 1; cool = now+self.policy.cooldown_seconds
            elif gap >= self.policy.heartbeat_seconds:
                count = min(count+1, self.policy.recovery_checks)
            else:
                return {'server_time': now, 'healthy_count': count}
            self.sql(tx, 'UPDATE @remote_nodes SET last_seen=?,healthy_count=?,cool_until=? WHERE id=?',
                     (now, count, cool, node['id']))
            return {'server_time': now, 'healthy_count': count}

    def healthy(self, node, now):
        return (not node['revoked'] and node['token_until'] > now
                and now-node['last_seen'] <= self.policy.heartbeat_seconds*3
                and node['healthy_count'] >= self.policy.recovery_checks and now >= node['cool_until'])

    def enroll_job(self, job_id, *, runtime_hash, memory_bytes, max_output_bytes, max_attempts=3):
        """Server admission only; does not alter immutable snapshot or deadline."""
        import re
        if (not re.fullmatch('[0-9a-f]{64}', runtime_hash) or type(max_attempts) is not int
                or not 1 <= max_attempts <= 10 or type(memory_bytes) is not int or memory_bytes <= 0
                or type(max_output_bytes) is not int or not 0 < max_output_bytes <= 1_000_000_000):
            raise ValueError('invalid remote job budget')
        with self.transaction() as tx:
            job = self.store._row(tx, job_id)
            if not job or job['state'] != 'queued': raise JobError('job_not_queued')
            snapshot = json.loads(job['snapshot'])
            if snapshot['operation'] not in OPERATIONS: raise JobError('operation_not_remote')
            self.files.validate_inputs(tx, job)
            count = self.sql(tx, '''SELECT count(*) AS n FROM @remote_jobs r JOIN {jobs} j ON j.id=r.job_id
                WHERE j.state IN ('queued','running','cancel_requested')''').fetchone()['n']
            if count >= self.policy.total_active: raise JobError('remote_queue_full', 429)
            self.sql(tx, '''INSERT INTO @remote_jobs(job_id,runtime_hash,memory_bytes,max_output_bytes,max_attempts)
                VALUES(?,?,?,?,?)''', (job_id, runtime_hash, memory_bytes, max_output_bytes, max_attempts))

    def _end(self, tx, attempt, job, code, retry):
        now = tx.now()
        count = self.sql(tx, 'SELECT count(*) AS n FROM @remote_attempts WHERE job_id=?', (job['id'],)).fetchone()['n']
        rule = self.sql(tx, 'SELECT * FROM @remote_jobs WHERE job_id=?', (job['id'],)).fetchone()
        state = ('queued' if retry and count < rule['max_attempts'] and now < job['deadline'] else 'failed')
        if job['state'] == 'cancel_requested' or code == 'cancelled': state = 'cancelled'
        self.sql(tx, 'UPDATE @remote_attempts SET ended_at=?,error_code=? WHERE id=?', (now, code, attempt['id']))
        tx.execute('UPDATE {jobs} SET state=?,error_code=?,worker_id=NULL,lease_until=NULL WHERE id=?',
                   (state, code, job['id']))
        self.sql(tx, "UPDATE @remote_jobs SET reason='taking_over' WHERE job_id=?", (job['id'],))
        job['state'] = state; self.store._event(tx, job, code, now)
        if attempt['node_id'] and retry:
            self.sql(tx, 'UPDATE @remote_nodes SET healthy_count=0,cool_until=? WHERE id=?',
                     (now+self.policy.cooldown_seconds, attempt['node_id']))
        # No quota release: failed attempt bytes/reservations await P07 physical cleanup.

    def _recover(self, tx):
        now = tx.now()
        attempts = self.sql(tx, 'SELECT * FROM @remote_attempts WHERE ended_at IS NULL').fetchall()
        for item in attempts:
            a = dict(item); j = self.store._row(tx, a['job_id'])
            if j['state'] not in ('running', 'cancel_requested') or j['generation'] != a['generation']:
                self.sql(tx, 'UPDATE @remote_attempts SET ended_at=?,error_code=? WHERE id=?', (now, 'externally_fenced', a['id']))
                continue
            code = None
            if j['state'] == 'cancel_requested': code = 'cancelled'
            elif j['deadline'] <= now: code = 'deadline_exceeded'
            elif j['lease_until'] <= now: code = 'lease_expired'
            elif a['node_id'] and a['phase'] in ('download', 'upload') and now-a['progress_at'] >= self.policy.transfer_seconds:
                code = 'transfer_stalled'
            if code: self._end(tx, a, j, code, code in NETWORK_ERRORS)
        for row in self.sql(tx, '''SELECT j.* FROM {jobs} j JOIN @remote_jobs r ON j.id=r.job_id
                WHERE j.state='queued' AND j.deadline<=?''', (now,)).fetchall():
            j = dict(row)
            tx.execute("UPDATE {jobs} SET state='failed',error_code='deadline_exceeded' WHERE id=?", (j['id'],))
            j['state'] = 'failed'; self.store._event(tx, j, 'deadline_exceeded', now)

    def recover(self):
        with self.transaction() as tx: self._recover(tx)

    def revoke(self, node_id):
        with self.transaction() as tx:
            self.sql(tx, 'UPDATE @remote_nodes SET revoked=1 WHERE id=?', (node_id,))
            for row in self.sql(tx, 'SELECT * FROM @remote_attempts WHERE node_id=? AND ended_at IS NULL', (node_id,)).fetchall():
                self._end(tx, dict(row), self.store._row(tx, row['job_id']), 'node_revoked', False)

    def _matches(self, tx, node, job, rule):
        try: self.scope(node, job)
        except JobError: return False
        count = self.sql(tx, 'SELECT count(*) AS n FROM @remote_attempts WHERE node_id=? AND ended_at IS NULL', (node['id'],)).fetchone()['n']
        memory = self.sql(tx, '''SELECT COALESCE(sum(r.memory_bytes),0) AS bytes FROM @remote_attempts a
            JOIN @remote_jobs r ON a.job_id=r.job_id WHERE a.node_id=? AND a.ended_at IS NULL''', (node['id'],)).fetchone()['bytes']
        return (self.healthy(node, tx.now()) and count < node['slots']
                and json.loads(job['snapshot'])['operation'] in json.loads(node['operations'])
                and node['runtime_hash'] == rule['runtime_hash'] and node['memory_bytes'] >= memory+rule['memory_bytes'])

    def claim(self, token, request_id, runtime_hash):
        with self.transaction() as tx:
            node = self.auth(tx, token)
            if runtime_hash != node['runtime_hash']: raise JobError('runtime_mismatch')
            self._recover(tx)
            old = self.sql(tx, 'SELECT * FROM @remote_attempts WHERE location=? AND request_id=?', (node['id'], str(request_id))).fetchone()
            if old:
                if old['ended_at'] is not None: return None
                self._fence(tx, token, old['id'], old['generation'])
                return self._manifest(tx, dict(old))
            return self._claim(tx, node, str(request_id))

    def claim_server(self):
        """Internal worker call; execution still MUST acquire P11 local Admission."""
        with self.transaction() as tx:
            self._recover(tx)
            return self._claim(tx, None, str(uuid4()))

    def _claim(self, tx, node, request_id):
        now = tx.now()
        nodes = [dict(n) for n in self.sql(tx, 'SELECT * FROM @remote_nodes').fetchall()]
        for value in self.sql(tx, '''SELECT j.* FROM {jobs} j JOIN @remote_jobs r ON j.id=r.job_id
                WHERE j.state='queued' ORDER BY j.created_at,j.id''').fetchall():
            j = dict(value); rule = dict(self.sql(tx, 'SELECT * FROM @remote_jobs WHERE job_id=?', (j['id'],)).fetchone())
            if tx.execute("SELECT count(*) AS n FROM {jobs} WHERE owner_id=? AND state IN ('running','cancel_requested')", (j['owner_id'],)).fetchone()['n'] >= self.policy.account_running:
                self._reason(tx, j, 'account_busy'); continue
            snapshot = json.loads(j['snapshot'])
            runtime_hash = rule['runtime_hash']
            if node:
                if not self._matches(tx, node, j, rule): continue
            else:
                if now-j['created_at'] < self.policy.node_wait_seconds and any(self._matches(tx, n, j, rule) for n in nodes):
                    self._reason(tx, j, 'waiting_node'); continue
                runtime_hash = self.server_capability(snapshot['operation'], rule['runtime_hash'])
                import re
                if not isinstance(runtime_hash, str) or not re.fullmatch('[0-9a-f]{64}', runtime_hash):
                    self._reason(tx, j, 'server_capability_unavailable'); continue
                if rule['memory_bytes'] > self.policy.server_memory_bytes:
                    self._reason(tx, j, 'server_over_budget'); continue
                if self.sql(tx, """SELECT count(*) AS n FROM {jobs} j WHERE j.state IN ('running','cancel_requested')
                    AND NOT EXISTS (SELECT 1 FROM @remote_attempts a WHERE a.job_id=j.id
                    AND a.generation=j.generation AND a.node_id IS NOT NULL AND a.ended_at IS NULL)""").fetchone()['n']:
                    self._reason(tx, j, 'server_busy'); continue
            try: inputs = self.files.validate_inputs(tx, j)
            except JobError as error:
                tx.execute("UPDATE {jobs} SET state='failed',error_code=? WHERE id=?", (error.code, j['id']))
                j['state'] = 'failed'; self.store._event(tx, j, error.code, now); continue
            aid = str(uuid4()); location = node['id'] if node else 'server'
            worker = 'remote:'+aid; gen = j['generation']+1
            lease = min(now+(self.policy.lease_seconds if node else self.store.lease_seconds), j['deadline'])
            if node:
                lease = min(lease, node['token_until'], *(i['expires_at'] for i in inputs))
            tx.execute("UPDATE {jobs} SET state='running',worker_id=?,generation=?,lease_until=?,error_code=NULL WHERE id=?", (worker, gen, lease, j['id']))
            self.sql(tx, '''INSERT INTO @remote_attempts(id,job_id,generation,node_id,location,request_id,worker_id,
                runtime_hash,parameter_hash,font_hash,input_hash,phase,started_at,progress_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (aid, j['id'], gen, node['id'] if node else None, location,
                request_id, worker, runtime_hash, digest(snapshot['config']), digest(snapshot['config'].get('analysis', {}).get('font')),
                digest(inputs), 'download' if node else 'compute', now, now))
            j['state'] = 'running'; self.store._event(tx, j, 'running_'+location if location == 'server' else 'running_node', now)
            return self._manifest(tx, dict(self.sql(tx, 'SELECT * FROM @remote_attempts WHERE id=?', (aid,)).fetchone()))
        return None

    def _reason(self, tx, job, reason):
        self.sql(tx, 'UPDATE @remote_jobs SET reason=? WHERE job_id=?', (reason, job['id']))

    def _manifest(self, tx, a):
        from ptb_api.remote_models import Snapshot, Claim
        j = self.store._row(tx, a['job_id']); s = json.loads(j['snapshot'])
        r = self.sql(tx, 'SELECT * FROM @remote_jobs WHERE job_id=?', (j['id'],)).fetchone()
        # Project/server-only snapshot metadata is deliberately not sent to nodes.
        snapshot = Snapshot.model_validate({k: s[k] for k in ('operation','core_version','adapter_version','config','input_refs')}).model_dump(mode='json')
        now = tx.now()
        inputs = [dict(i, expires_remaining_seconds=i['expires_at']-now) for i in self.files.validate_inputs(tx, j)]
        value = dict(protocol='remote/1', attempt_id=a['id'], job_id=j['id'], generation=a['generation'],
            **self._lease(j, now), location=a['location'], operation=s['operation'],
            runtime_hash=a['runtime_hash'], parameter_hash=a['parameter_hash'], font_hash=a['font_hash'],
            input_hash=a['input_hash'], snapshot=snapshot, inputs=inputs, memory_bytes=r['memory_bytes'], max_output_bytes=r['max_output_bytes'])
        return Claim.model_validate(value).model_dump(mode='json')

    @staticmethod
    def _lease(job, now):
        return dict(server_time=now, lease_until=job['lease_until'], deadline=job['deadline'],
                    lease_remaining_seconds=job['lease_until']-now, deadline_remaining_seconds=job['deadline']-now)

    def _fence(self, tx, token, aid, generation, *, receipt=False):
        node = self.auth(tx, token)
        row = self.sql(tx, 'SELECT * FROM @remote_attempts WHERE id=?', (str(aid),)).fetchone()
        if not row or row['node_id'] != node['id'] or row['generation'] != generation:
            raise JobError('stale_attempt')
        a = dict(row); j = self.store._row(tx, a['job_id']); self.scope(node, j)
        if j['generation'] != generation: raise JobError('stale_attempt')
        if receipt and a['receipt'] and j['state'] == 'succeeded': return a, j
        if j['state'] == 'cancel_requested': raise JobError('cancelled')
        if (a['ended_at'] is not None or j['state'] != 'running' or j['worker_id'] != a['worker_id']
                or j['lease_until'] <= tx.now()): raise JobError('stale_attempt')
        if j['deadline'] <= tx.now(): raise JobError('deadline_exceeded', 410)
        if a['phase'] in ('download', 'upload') and tx.now()-a['progress_at'] >= self.policy.transfer_seconds:
            raise JobError('transfer_stalled')
        self.files.validate_inputs(tx, j)
        return a, j

    def heartbeat(self, token, aid, generation, phase, node_bytes=0):
        if phase not in ('download', 'compute', 'upload') or type(node_bytes) is not int or not 0 <= node_bytes <= 1_000_000_000_000:
            raise JobError('invalid_request', 422)
        with self.transaction() as tx:
            a, j = self._fence(tx, token, aid, generation)
            stages = {'download': 0, 'compute': 1, 'upload': 2}
            if not 0 <= stages[phase]-stages[a['phase']] <= 1: raise JobError('invalid_phase')
            if phase == 'compute' and a['phase'] == 'download':
                for item in self.files.validate_inputs(tx, j):
                    read = self.sql(tx, 'SELECT offset_bytes FROM @remote_reads WHERE attempt_id=? AND asset_id=?', (str(aid), item['id'])).fetchone()
                    if not read or read['offset_bytes'] != item['size_bytes']: raise JobError('input_download_incomplete')
            inputs = self.files.validate_inputs(tx, j)
            node = self.auth(tx, token)
            until = min(tx.now()+self.policy.lease_seconds, j['deadline'], node['token_until'],
                        *(i['expires_at'] for i in inputs))
            tx.execute('UPDATE {jobs} SET lease_until=? WHERE id=?', (until, j['id']))
            self.sql(tx, 'UPDATE @remote_attempts SET phase=?,node_bytes=?,progress_at=? WHERE id=?',
                     (phase, node_bytes, tx.now() if phase != a['phase'] else a['progress_at'], str(aid)))
            return self._lease(dict(j, lease_until=until), tx.now())

    def check(self, token, aid, generation):
        with self.transaction() as tx:
            self._fence(tx, token, aid, generation)

    def read(self, token, aid, generation, asset_id, offset, size):
        if type(offset) is not int or offset < 0 or type(size) is not int or not 0 < size <= CHUNK_BYTES:
            raise JobError('invalid_chunk', 413)
        with self.transaction() as tx:
            a, j = self._fence(tx, token, aid, generation)
            inputs = self.files.validate_inputs(tx, j)
            if asset_id not in {x['id'] for x in inputs}: raise JobError('asset_not_found', 404)
            old = self.sql(tx, 'SELECT offset_bytes FROM @remote_reads WHERE attempt_id=? AND asset_id=?', (str(aid), asset_id)).fetchone()
            end = old['offset_bytes'] if old else 0
            if offset > end: raise JobError('chunk_offset')
            data = self.files.read(tx, j, asset_id, offset, size)
            self._fence(tx, token, aid, generation)
            if offset+len(data) > end:
                self.sql(tx, '''INSERT INTO @remote_reads(attempt_id,asset_id,offset_bytes) VALUES(?,?,?)
                    ON CONFLICT(attempt_id,asset_id) DO UPDATE SET offset_bytes=excluded.offset_bytes''', (str(aid), asset_id, offset+len(data)))
                self.sql(tx, 'UPDATE @remote_attempts SET progress_at=? WHERE id=?', (tx.now(), str(aid)))
            return data

    def output(self, token, aid, body):
        with self.transaction() as tx:
            a, j = self._fence(tx, token, aid, body.generation)
            if a['phase'] != 'upload': raise JobError('invalid_phase')
            old = self.sql(tx, 'SELECT * FROM @remote_uploads WHERE attempt_id=? AND output_key=?', (str(aid), body.key)).fetchone()
            if old:
                if any(old[k] != getattr(body, k) for k in ('name', 'size_bytes', 'sha256')): raise JobError('idempotency_conflict')
                return {'upload_id': old['id'], 'offset': old['offset_bytes']}
            totals = self.sql(tx, 'SELECT count(*) AS n,COALESCE(sum(size_bytes),0) AS bytes FROM @remote_uploads WHERE attempt_id=?', (str(aid),)).fetchone()
            rule = self.sql(tx, 'SELECT * FROM @remote_jobs WHERE job_id=?', (j['id'],)).fetchone()
            if totals['n'] >= 16 or totals['bytes']+body.size_bytes > rule['max_output_bytes']: raise JobError('output_budget_exceeded', 413)
            uid = str(uuid4()); self.files.reserve(tx, j, uid, body.name, body.size_bytes)
            self._fence(tx, token, aid, body.generation)
            self.sql(tx, 'INSERT INTO @remote_uploads(id,attempt_id,output_key,name,size_bytes,sha256) VALUES(?,?,?,?,?,?)',
                     (uid, str(aid), body.key, body.name, body.size_bytes, body.sha256))
            return {'upload_id': uid, 'offset': 0}

    def write(self, token, aid, generation, uid, offset, data, sha256):
        if type(offset) is not int or offset < 0 or not 0 < len(data) <= CHUNK_BYTES: raise JobError('invalid_chunk', 413)
        if hashlib.sha256(data).hexdigest() != sha256: raise JobError('hash_mismatch', 422)
        with self.transaction() as tx:
            a, j = self._fence(tx, token, aid, generation)
            u = self.sql(tx, 'SELECT * FROM @remote_uploads WHERE id=? AND attempt_id=?', (str(uid), str(aid))).fetchone()
            if not u: raise JobError('upload_not_found', 404)
            old = self.sql(tx, 'SELECT * FROM @remote_chunks WHERE upload_id=? AND offset_bytes=?', (str(uid), offset)).fetchone()
            if old:
                if old['sha256'] != sha256 or old['size_bytes'] != len(data): raise JobError('idempotency_conflict')
                return {'offset': u['offset_bytes']}
            if offset != u['offset_bytes'] or offset+len(data) > u['size_bytes']: raise JobError('chunk_offset')
            self.files.append(tx, j, str(uid), offset, data)
            self._fence(tx, token, aid, generation)
            self.sql(tx, 'INSERT INTO @remote_chunks(upload_id,offset_bytes,size_bytes,sha256) VALUES(?,?,?,?)', (str(uid), offset, len(data), sha256))
            self.sql(tx, 'UPDATE @remote_uploads SET offset_bytes=? WHERE id=?', (offset+len(data), str(uid)))
            self.sql(tx, 'UPDATE @remote_attempts SET progress_at=? WHERE id=?', (tx.now(), str(aid)))
            return {'offset': offset+len(data)}

    def complete(self, token, aid, generation):
        with self.transaction() as tx:
            a, j = self._fence(tx, token, aid, generation, receipt=True)
            if a['receipt']: return json.loads(a['receipt'])
            uploads = [dict(u) for u in self.sql(tx, 'SELECT * FROM @remote_uploads WHERE attempt_id=? ORDER BY output_key', (str(aid),)).fetchall()]
            if not uploads or any(u['offset_bytes'] != u['size_bytes'] for u in uploads): raise JobError('incomplete_outputs')
            for u in uploads: self.files.seal(tx, j, u['id'], u['sha256'])
            manifest = self.files.publish(tx, j, uploads)
            self._fence(tx, token, aid, generation)
            encoded = canonical(manifest)
            tx.execute("UPDATE {jobs} SET state='succeeded',result_manifest=?,progress=1,worker_id=NULL,lease_until=NULL,error_code=NULL WHERE id=?", (encoded, j['id']))
            self.sql(tx, 'UPDATE @remote_attempts SET ended_at=?,receipt=? WHERE id=?', (tx.now(), encoded, str(aid)))
            j.update(state='succeeded', progress=1); self.store._event(tx, j, 'succeeded', tx.now())
            return manifest

    def fail(self, token, aid, generation, code):
        from ptb_api.remote_models import Failure
        Failure(generation=generation, code=code)
        with self.transaction() as tx:
            a, j = self._fence(tx, token, aid, generation)
            self._end(tx, a, j, code, code in RETRYABLE_ERRORS)
            return {'accepted': True}
