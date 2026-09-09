"""P05 PostgreSQL adapter. Construction never creates or migrates a database."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Protocol
from uuid import uuid4
import re

import psycopg
from psycopg.rows import dict_row
from argon2 import PasswordHasher
from argon2.low_level import Type

PASSWORDS = PasswordHasher(type=Type.ID, time_cost=3, memory_cost=65536, parallelism=4)

def hash_token(value: str) -> str:
    return sha256(value.encode('utf-8')).hexdigest()

class AccountStore(Protocol):
    def get_user(self, username: str): ...
    def allow_login(self, username: str, ip: str) -> bool: ...
    def issue_session(self, user: dict, token_hash: str, csrf: str, expires, old_hash: str): ...
    def session(self, token_hash: str): ...
    def revoke(self, token_hash: str): ...
    def list_projects(self, owner: str): ...
    def create_project(self, owner: str, name: str): ...
    def get_project(self, owner: str, project_id: str): ...
    def rename_project(self, owner: str, project_id: str, name: str): ...

class PostgresAccountStore:
    def __init__(self, dsn: str):
        self._dsn = dsn

    @contextmanager
    def connection(self):
        with psycopg.connect(self._dsn, row_factory=dict_row, connect_timeout=5,
                             options='-c statement_timeout=10000 -c lock_timeout=5000') as conn:
            yield conn

    def check_schema(self):
        with self.connection() as c:
            if c.execute('SELECT version FROM ptb_accounts.schema_version').fetchall() != [{'version': 1}]:
                raise RuntimeError('P05 schema version mismatch')

    def create_user(self, username: str, password: str):
        username = username.lower()
        if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]{2,63}', username) or not 12 <= len(password) <= 1024:
            raise ValueError('Use a 3–64 character login and a 12–1024 character password')
        digest = PASSWORDS.hash(password)
        with self.connection() as c:
            return c.execute('INSERT INTO ptb_accounts.users(id,username,password_hash) VALUES (%s,%s,%s) RETURNING id,username',
                             (uuid4(), username, digest)).fetchone()

    def get_user(self, username):
        with self.connection() as c:
            return c.execute('SELECT id,username,password_hash,active FROM ptb_accounts.users WHERE username=%s', (username,)).fetchone()

    def allow_login(self, username, ip):
        # Fixed-window attempt reservation is atomic across connections/processes.
        limits = [(hash_token('account:'+username), 10), (hash_token('ip:'+ip), 50)]
        allowed = True
        with self.connection() as c:
            for key, limit in sorted(limits):
                row = c.execute("""INSERT INTO ptb_accounts.login_attempts AS t(bucket_key,attempts,window_start)
                    VALUES (%s,1,clock_timestamp()) ON CONFLICT(bucket_key) DO UPDATE SET
                    attempts=CASE WHEN t.window_start <= clock_timestamp()-interval '15 minutes' THEN 1 ELSE t.attempts+1 END,
                    window_start=CASE WHEN t.window_start <= clock_timestamp()-interval '15 minutes' THEN clock_timestamp() ELSE t.window_start END
                    RETURNING attempts""", (key,)).fetchone()
                allowed = allowed and row['attempts'] <= limit
        return allowed

    def issue_session(self, user, token_hash, csrf, expires, old_hash):
        with self.connection() as c:
            row = c.execute('SELECT id FROM ptb_accounts.users WHERE id=%s AND active AND password_hash=%s FOR UPDATE',
                            (user['id'], user['password_hash'])).fetchone()
            if not row:
                return False
            c.execute('UPDATE ptb_accounts.sessions SET revoked_at=now() WHERE token_hash=%s AND revoked_at IS NULL', (old_hash,))
            c.execute('INSERT INTO ptb_accounts.sessions(token_hash,user_id,csrf_token,expires_at) VALUES (%s,%s,%s,%s)',
                      (token_hash, user['id'], csrf, expires))
            return True

    def session(self, token_hash):
        with self.connection() as c:
            row = c.execute("""SELECT s.csrf_token,s.expires_at,u.id,u.username FROM ptb_accounts.sessions s
                JOIN ptb_accounts.users u ON u.id=s.user_id
                WHERE s.token_hash=%s AND s.revoked_at IS NULL AND s.expires_at>now() AND u.active""", (token_hash,)).fetchone()
            if row:
                row['id'] = str(row['id'])
            return row

    def revoke(self, token_hash):
        with self.connection() as c:
            c.execute('UPDATE ptb_accounts.sessions SET revoked_at=now() WHERE token_hash=%s', (token_hash,))

    @staticmethod
    def project(row):
        if row:
            row['id'] = str(row['id'])
        return row

    def list_projects(self, owner):
        with self.connection() as c:
            return [self.project(row) for row in c.execute(
                'SELECT id,name,created_at FROM ptb_accounts.projects WHERE owner_id=%s ORDER BY created_at DESC,id DESC LIMIT 100', (owner,)).fetchall()]

    def create_project(self, owner, name):
        with self.connection() as c:
            # Per-owner serialization also bounds project metadata growth.
            c.execute('SELECT id FROM ptb_accounts.users WHERE id=%s AND active FOR UPDATE', (owner,))
            if c.execute('SELECT count(*) AS n FROM ptb_accounts.projects WHERE owner_id=%s', (owner,)).fetchone()['n'] >= 100:
                return None
            return self.project(c.execute('INSERT INTO ptb_accounts.projects(id,owner_id,name) VALUES (%s,%s,%s) RETURNING id,name,created_at',
                                          (uuid4(), owner, name)).fetchone())

    def get_project(self, owner, project_id):
        with self.connection() as c:
            return self.project(c.execute('SELECT id,name,created_at FROM ptb_accounts.projects WHERE id=%s AND owner_id=%s', (project_id, owner)).fetchone())

    def rename_project(self, owner, project_id, name):
        with self.connection() as c:
            return self.project(c.execute('UPDATE ptb_accounts.projects SET name=%s WHERE id=%s AND owner_id=%s RETURNING id,name,created_at', (name, project_id, owner)).fetchone())
