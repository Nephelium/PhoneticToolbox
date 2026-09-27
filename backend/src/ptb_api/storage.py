"""P07 private file adapter. Construction never initializes or deletes data.

All disk operations require the same OS lock in addition to the PG lock. A
committed reservation covers bytes between file fsync and ledger settlement.
This adapter accepts controlled blocks only, never a native-tool output path.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from .quota import CHUNK_BYTES, QUOTA_BYTES, TEMP_SECONDS, StorageError, reserve, expiry
from .storage_policy import POLICY_VERSION, LEGACY_POLICY_VERSION, LEGACY_QUOTA_BYTES, retention_seconds

LOCK_ID = 577707


def regular(path):
    info = path.lstat()
    return stat.S_ISREG(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400


def no_links(path):
    for part in [path, *path.parents]:
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise StorageError('storage_path_rejected', 503)


def public_asset(row):
    fields = ('name', 'kind', 'state', 'size_bytes', 'reserved_bytes', 'expected_bytes',
              'sha256', 'created_at', 'expires_at', 'error_code')
    return {key: row[key] for key in fields} | {'id': str(row['id']), 'project_id': str(row['project_id']),
        'policy_version': row.get('policy_version', LEGACY_POLICY_VERSION)}


class Storage:
    def __init__(self, dsn, root, *, min_free_bytes=1_000_000_000):
        if type(min_free_bytes) is not int or min_free_bytes < 0:
            raise ValueError('Invalid disk safety margin')
        self._dsn = dsn
        self.root = Path(os.path.abspath(root))
        self.min_free_bytes = min_free_bytes
        self.ready = False
        self.files = None

    def _path(self, asset_id):
        # No user filename, extension, directory, storage URL or shell input.
        name = str(UUID(str(asset_id))) + '.bin'
        path = self.root / name
        if path.exists() or path.is_symlink():
            if not regular(path):
                raise StorageError('storage_path_rejected', 503)
        return path

    @contextmanager
    def _locked(self):
        no_links(self.root)
        marker = self.root / '.ptb-storage.json'
        lock = self.root / '.ptb-storage.lock'
        if not regular(marker) or not regular(lock):
            raise StorageError('storage_not_initialized', 503)
        instance = json.loads(marker.read_text('utf-8'))['instance_id']
        with lock.open('r+b', buffering=0) as handle:
            deadline = time.monotonic()+5
            while True:
                try:
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise StorageError('storage_busy', 503) from None
                    time.sleep(0.01)
            try:
                with psycopg.connect(self._dsn, autocommit=True, row_factory=dict_row,
                                     connect_timeout=5, options='-c statement_timeout=10000 -c lock_timeout=5000') as conn:
                    if not conn.execute('SELECT pg_try_advisory_lock(%s) AS acquired', (LOCK_ID,)).fetchone()['acquired']:
                        raise StorageError('storage_busy', 503)
                    state = conn.execute('SELECT * FROM ptb_storage.state WHERE singleton').fetchone()
                    if not state or state['version'] != 1 or str(state['instance_id']) != instance:
                        raise StorageError('storage_instance_mismatch', 503)
                    yield conn
            finally:
                if os.name == 'nt':
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _now(conn):
        return float(conn.execute('SELECT extract(epoch FROM clock_timestamp()) AS t').fetchone()['t'])

    @staticmethod
    def _row(conn, owner, asset_id):
        row = conn.execute('SELECT * FROM ptb_storage.assets WHERE id=%s AND owner_id=%s',
                           (asset_id, owner)).fetchone()
        if not row:
            raise StorageError('asset_not_found', 404)
        return row

    @staticmethod
    def _freeze(conn, reason):
        conn.execute('UPDATE ptb_storage.state SET frozen=true,reason=%s WHERE singleton', (reason,))
        raise StorageError('storage_inconsistent', 503)

    def _writable(self, conn):
        state = conn.execute('SELECT * FROM ptb_storage.state').fetchone()
        if not self.ready or state['frozen']:
            raise StorageError('storage_recovery_required', 503)
        if state.get('policy_version', LEGACY_POLICY_VERSION) != POLICY_VERSION:
            raise StorageError('storage_policy_migration_required', 503)

    def _disk_budget(self, conn, extra):
        reserved = conn.execute('SELECT coalesce(sum(reserved_bytes),0) AS n FROM ptb_storage.quota_accounts').fetchone()['n']
        if shutil.disk_usage(self.root).free - reserved - extra < self.min_free_bytes:
            raise StorageError('disk_space_low', 507)

    def _sync(self, conn, row):
        path = self._path(row['id'])
        if not path.exists() and row['state'] == 'uploading' and row['size_bytes'] == 0:
            with path.open('xb') as target:
                target.flush()
                os.fsync(target.fileno())
        actual = path.stat().st_size if path.exists() else 0
        if not path.exists() and row['state'] == 'ready':
            self._freeze(conn, 'missing_ready_file')
        delta = actual - row['size_bytes']
        if delta < 0 or delta > row['reserved_bytes']:
            self._freeze(conn, 'file_size_mismatch')
        if delta:
            with conn.transaction():
                conn.execute('UPDATE ptb_storage.assets SET size_bytes=size_bytes+%s,reserved_bytes=reserved_bytes-%s WHERE id=%s',
                             (delta, delta, row['id']))
                conn.execute('UPDATE ptb_storage.quota_accounts SET used_bytes=used_bytes+%s,reserved_bytes=reserved_bytes-%s WHERE owner_id=%s',
                             (delta, delta, row['owner_id']))
        return self._row(conn, row['owner_id'], row['id'])

    def usage(self, owner):
        with self._locked() as conn:
            row = conn.execute('SELECT * FROM ptb_storage.quota_accounts WHERE owner_id=%s', (owner,)).fetchone()
            used, reserved = (row['used_bytes'], row['reserved_bytes']) if row else (0, 0)
            state = conn.execute('SELECT * FROM ptb_storage.state').fetchone()
            version = state.get('policy_version', LEGACY_POLICY_VERSION)
            quota = row['quota_bytes'] if row else (QUOTA_BYTES if version == POLICY_VERSION else LEGACY_QUOTA_BYTES)
            return dict(quota_bytes=quota, used_bytes=used, reserved_bytes=reserved,
                        available_bytes=max(0, quota-used-reserved), frozen=state['frozen'], ready=self.ready,
                        policy_version=version, retention_seconds=retention_seconds(version),
                        over_quota=used+reserved>quota)

    def list(self, owner, project_id, order='expires'):
        column = {'expires': 'expires_at,id', 'size': 'size_bytes DESC,id', 'created': 'created_at DESC,id'}[order]
        with self._locked() as conn:
            return [public_asset(r) for r in conn.execute(
                f"SELECT * FROM ptb_storage.assets WHERE owner_id=%s AND project_id=%s AND state!='deleted' ORDER BY {column} LIMIT 1000",
                (owner, project_id)).fetchall()]

    def create(self, owner, body):
        with self._locked() as conn:
            return self._create(conn, owner, body)

    def _create(self, conn, owner, body, *, kind="input", deadline=None, on_created=None):
        payload = body.model_dump(mode='json', exclude={'idempotency_key'})
        if kind != 'input':
            payload['kind'] = kind
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        old = conn.execute('SELECT * FROM ptb_storage.assets WHERE owner_id=%s AND idempotency_key=%s',
                           (owner, body.idempotency_key)).fetchone()
        if old:
            if old['request_hash'] != digest:
                raise StorageError('idempotency_conflict')
            return public_asset(old)
        self._writable(conn)
        if not conn.execute('SELECT 1 FROM ptb_accounts.projects p JOIN ptb_accounts.users u ON p.owner_id=u.id WHERE p.id=%s AND p.owner_id=%s AND u.active',
                            (body.project_id, owner)).fetchone():
            raise StorageError('project_not_found', 404)
        if conn.execute('SELECT count(*) AS n FROM ptb_storage.assets WHERE owner_id=%s', (owner,)).fetchone()['n'] >= 10000:
            raise StorageError('asset_limit_reached')
        budget = body.expected_bytes or 0
        self._disk_budget(conn, budget)
        now, asset_id = self._now(conn), uuid4()
        with conn.transaction():
            conn.execute('INSERT INTO ptb_storage.quota_accounts(owner_id) VALUES(%s) ON CONFLICT DO NOTHING', (owner,))
            q = conn.execute('SELECT * FROM ptb_storage.quota_accounts WHERE owner_id=%s FOR UPDATE', (owner,)).fetchone()
            reserve(q['used_bytes'], q['reserved_bytes'], budget)
            conn.execute('UPDATE ptb_storage.quota_accounts SET reserved_bytes=reserved_bytes+%s WHERE owner_id=%s', (budget, owner))
            conn.execute("INSERT INTO ptb_storage.assets(id,owner_id,project_id,name,state,idempotency_key,request_hash,expected_bytes,reserved_bytes,created_at,expires_at) VALUES(%s,%s,%s,%s,'uploading',%s,%s,%s,%s,%s,%s)",
                         (asset_id, owner, body.project_id, body.name, body.idempotency_key, digest, body.expected_bytes, budget, now, min(now+TEMP_SECONDS, deadline) if deadline else now+TEMP_SECONDS))
            if kind != "input":
                conn.execute("UPDATE ptb_storage.assets SET kind=%s WHERE id=%s", (kind, asset_id))
            if on_created:
                on_created(conn, asset_id)
        # The durable row/reservation exists before even an empty file is created.
        with self._path(asset_id).open('xb') as target:
            target.flush()
            os.fsync(target.fileno())
        return public_asset(self._row(conn, owner, asset_id))

    def append(self, owner, asset_id, offset, data):
        with self._locked() as conn:
            if self._row(conn, owner, asset_id)["kind"] != "input":
                raise StorageError("worker_owned_asset", 403)
            return self._append(conn, owner, asset_id, offset, data)

    def _append(self, conn, owner, asset_id, offset, data):
        if not 0 < len(data) <= CHUNK_BYTES or type(offset) is not int or offset < 0:
            raise StorageError('invalid_chunk', 422)
        self._writable(conn)
        row = self._sync(conn, self._row(conn, owner, asset_id))
        if row['state'] != 'uploading' or row['expires_at'] <= self._now(conn):
            raise StorageError('upload_closed', 410)
        path = self._path(asset_id)
        if offset < row['size_bytes'] and offset+len(data) <= row['size_bytes']:
            with path.open('rb') as source:
                source.seek(offset)
                if source.read(len(data)) == data:
                    return public_asset(row)
            raise StorageError('chunk_conflict')
        if offset != row['size_bytes']:
            raise StorageError('offset_conflict')
        if row['expected_bytes'] is not None and offset+len(data) > row['expected_bytes']:
            raise StorageError('declared_size_exceeded', 413)
        extra = max(0, len(data)-row['reserved_bytes'])
        self._disk_budget(conn, extra)
        with conn.transaction():
            q = conn.execute('SELECT * FROM ptb_storage.quota_accounts WHERE owner_id=%s FOR UPDATE', (owner,)).fetchone()
            reserve(q['used_bytes'], q['reserved_bytes'], extra)
            conn.execute('UPDATE ptb_storage.assets SET reserved_bytes=reserved_bytes+%s WHERE id=%s', (extra, asset_id))
            conn.execute('UPDATE ptb_storage.quota_accounts SET reserved_bytes=reserved_bytes+%s WHERE owner_id=%s', (extra, owner))
        try:
            # No truncate or overwrite; missing empty files can be recovered from
            # a crash between committed reservation and exclusive creation.
            with path.open('ab', buffering=0) as target:
                written = target.write(data)
                if written != len(data):
                    raise OSError('Incomplete storage write')
                os.fsync(target.fileno())
        except OSError:
            # Reservation remains held. Recovery accounts even partially written bytes.
            raise StorageError('storage_write_failed', 507) from None
        return public_asset(self._sync(conn, self._row(conn, owner, asset_id)))

    def finalize(self, owner, asset_id, expected_hash=None):
        with self._locked() as conn:
            if self._row(conn, owner, asset_id)["kind"] != "input":
                raise StorageError("worker_owned_asset", 403)
            self._writable(conn)
            row = self._sync(conn, self._row(conn, owner, asset_id))
            if row['state'] == 'ready':
                if expected_hash is not None and row['sha256'] != expected_hash:
                    raise StorageError('checksum_mismatch')
                if row['expires_at'] <= self._now(conn):
                    raise StorageError('asset_expired', 410)
                return public_asset(row)
            if row['state'] != 'uploading' or row['expires_at'] <= self._now(conn):
                raise StorageError('upload_closed', 410)
            if row['expected_bytes'] is not None and row['size_bytes'] != row['expected_bytes']:
                raise StorageError('upload_incomplete')
            digest = hashlib.sha256()
            with self._path(asset_id).open('rb') as source:
                for block in iter(lambda: source.read(CHUNK_BYTES), b''):
                    digest.update(block)
            computed_hash = digest.hexdigest()
            if expected_hash is not None and computed_hash != expected_hash:
                raise StorageError('checksum_mismatch')
            now = self._now(conn)
            if now >= row['expires_at']:
                raise StorageError('upload_closed', 410)
            with conn.transaction():
                conn.execute('UPDATE ptb_storage.quota_accounts SET reserved_bytes=reserved_bytes-%s WHERE owner_id=%s', (row['reserved_bytes'], owner))
                conn.execute("UPDATE ptb_storage.assets SET state='ready',reserved_bytes=0,sha256=%s,expires_at=%s,policy_version=%s WHERE id=%s",
                             (computed_hash, expiry(now), POLICY_VERSION, asset_id))
            return public_asset(self._row(conn, owner, asset_id))

    def _readable(self, conn, owner, asset_id):
        if not self.ready:
            raise StorageError('storage_recovery_required', 503)
        row = self._row(conn, owner, asset_id)
        if row['state'] != 'ready' or row['expires_at'] <= self._now(conn):
            raise StorageError('asset_expired' if row['expires_at'] <= self._now(conn) else 'asset_unavailable', 410)
        path = self._path(asset_id)
        if not path.exists() or path.stat().st_size != row['size_bytes']:
            self._freeze(conn, 'file_size_mismatch')
        return row

    def metadata(self, owner, asset_id):
        with self._locked() as conn:
            return public_asset(self._readable(conn, owner, asset_id))

    def read_block(self, owner, asset_id, offset, size):
        if not 0 <= offset or not 0 < size <= CHUNK_BYTES:
            raise StorageError('invalid_chunk', 422)
        with self._locked() as conn:
            row = self._readable(conn, owner, asset_id)
            with self._path(asset_id).open('rb') as source:
                source.seek(offset)
                block = source.read(min(size, max(0, row['size_bytes']-offset)))
            if self._now(conn) >= row['expires_at']:
                raise StorageError('asset_expired', 410)
            return block

    def _delete(self, conn, row, *, notify_jobs=True):
        if row['state'] == 'deleted':
            return row
        if notify_jobs and self.files is not None:
            self.files.before_delete(conn, row)
        # Incomplete disk writes must stay budget-covered until actual deletion.
        conn.execute("UPDATE ptb_storage.assets SET state='deleting',delete_attempts=delete_attempts+1,last_delete_at=%s WHERE id=%s",
                     (self._now(conn), row['id']))
        path = self._path(row['id'])
        try:
            path.unlink(missing_ok=True)
        except OSError:
            conn.execute("UPDATE ptb_storage.assets SET state='delete_failed',error_code='physical_delete_failed' WHERE id=%s", (row['id'],))
            return self._row(conn, row['owner_id'], row['id'])
        with conn.transaction():
            conn.execute('UPDATE ptb_storage.quota_accounts SET used_bytes=used_bytes-%s,reserved_bytes=reserved_bytes-%s WHERE owner_id=%s',
                         (row['size_bytes'], row['reserved_bytes'], row['owner_id']))
            conn.execute("UPDATE ptb_storage.assets SET state='deleted',name='[deleted]',size_bytes=0,reserved_bytes=0,sha256=NULL,error_code=NULL,deleted_at=%s WHERE id=%s",
                         (self._now(conn), row['id']))
        return self._row(conn, row['owner_id'], row['id'])

    def delete(self, owner, asset_id):
        with self._locked() as conn:
            return public_asset(self._delete(conn, self._row(conn, owner, asset_id)))

    def impact(self, owner, asset_id):
        if self.files is not None:
            return self.files.impact(owner,asset_id)
        with self._locked() as conn:
            self._row(conn,owner,asset_id)
            return {'active_jobs':[]}

    def recover(self):
        """Called before downloads; unknown files freeze writes instead of disappearing."""
        self.ready = False
        with self._locked() as conn:
            conn.execute("UPDATE ptb_storage.state SET frozen=true,reason='recovering'")
            if self.files is not None:
                self.files.reconcile(conn)
            rows = conn.execute('SELECT * FROM ptb_storage.assets').fetchall()
            expected = {str(r['id'])+'.bin' for r in rows if r['state'] != 'deleted'}
            actual = {p.name for p in self.root.iterdir()} - {'.ptb-storage.json', '.ptb-storage.lock'}
            if actual - expected:
                self._freeze(conn, 'unknown_files')
            for row in rows:
                if row['state'] == 'deleted':
                    continue
                if row['state'] in ('deleting', 'delete_failed') or row['expires_at'] <= self._now(conn):
                    self._delete(conn, row)
                else:
                    self._sync(conn, row)
            # Reconcile the ledger itself before reopening writes.
            mismatch = conn.execute('''SELECT 1 FROM ptb_storage.quota_accounts q LEFT JOIN
                (SELECT owner_id,sum(size_bytes) AS used,sum(reserved_bytes) AS reserved FROM ptb_storage.assets GROUP BY owner_id) a
                ON a.owner_id=q.owner_id WHERE q.used_bytes!=coalesce(a.used,0) OR q.reserved_bytes!=coalesce(a.reserved,0)''').fetchone()
            if mismatch:
                self._freeze(conn, 'ledger_mismatch')
            conn.execute('UPDATE ptb_storage.state SET frozen=false,reason=NULL')
            self.ready = True

    def cleanup(self):
        with self._locked() as conn:
            if self.files is not None:
                self.files.reconcile(conn)
            now = self._now(conn)
            rows = conn.execute("SELECT * FROM ptb_storage.assets WHERE state!='deleted' AND (expires_at<=%s OR state IN ('deleting','delete_failed')) ORDER BY expires_at LIMIT 100", (now,)).fetchall()
            for row in rows:
                self._delete(conn, row)
            next_due = conn.execute("SELECT min(expires_at) AS t FROM ptb_storage.assets WHERE state IN ('uploading','ready') AND expires_at>%s", (now,)).fetchone()['t']
            return {'attempted': len(rows), 'next_due': next_due}
