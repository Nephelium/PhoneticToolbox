"""Email identity extension using the existing PostgreSQL account tables.

No automatic DDL, no plaintext token persistence, no feedback identity access.
"""
from uuid import uuid4
from .account_store import PostgresAccountStore, hash_token


class PostgresRegistrationStore(PostgresAccountStore):
    def check_schema(self):
        super().check_schema()
        with self.connection() as c:
            if c.execute('SELECT version FROM ptb_web.schema_version').fetchall() != [{'version': 1}]:
                raise RuntimeError('Web identity migration required')

    def allow_email_action(self, action, email, ip):
        allowed = True
        # Rate reservations commit even when the action is denied.
        limits = [(hash_token('web-mail:'+email), 5), (hash_token('web-ip:'+ip), 30)]
        with self.connection() as c:
            for key, limit in sorted(limits):
                row = c.execute("""INSERT INTO ptb_web.rate_limits AS t(bucket_key,attempts,window_start)
                    VALUES(%s,1,clock_timestamp()) ON CONFLICT(bucket_key) DO UPDATE SET
                    attempts=CASE WHEN t.window_start<=clock_timestamp()-interval '15 minutes' THEN 1 ELSE t.attempts+1 END,
                    window_start=CASE WHEN t.window_start<=clock_timestamp()-interval '15 minutes' THEN clock_timestamp() ELSE t.window_start END
                    RETURNING attempts""", (key,)).fetchone()
                allowed = allowed and row['attempts'] <= limit
            c.execute("DELETE FROM ptb_web.rate_limits WHERE window_start<clock_timestamp()-interval '1 day'")
            c.execute('DELETE FROM ptb_web.email_challenges WHERE expires_at<=clock_timestamp()')
        return allowed

    @staticmethod
    def _lock(c, *names):
        for name in sorted(names):
            c.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))', (name,))

    def begin_registration(self, username, email, password_hash, token_hash):
        with self.connection() as c:
            self._lock(c, 'email:'+email, 'username:'+username)
            if c.execute('SELECT 1 FROM ptb_accounts.users WHERE username=%s', (username,)).fetchone():
                return False
            if c.execute('SELECT 1 FROM ptb_web.emails WHERE email=%s', (email,)).fetchone():
                return False
            c.execute("DELETE FROM ptb_web.email_challenges WHERE email=%s AND purpose='register'", (email,))
            c.execute("""INSERT INTO ptb_web.email_challenges(token_hash,purpose,email,username,password_hash,expires_at)
                VALUES(%s,'register',%s,%s,%s,clock_timestamp()+interval '30 minutes')""",
                (token_hash,email,username,password_hash))
            return True

    def begin_recovery(self, email, token_hash):
        with self.connection() as c:
            self._lock(c, 'email:'+email)
            user = c.execute('''SELECT u.id FROM ptb_web.emails e JOIN ptb_accounts.users u ON u.id=e.user_id
                WHERE e.email=%s AND u.active''', (email,)).fetchone()
            if not user:
                return False
            c.execute("DELETE FROM ptb_web.email_challenges WHERE email=%s AND purpose='reset'", (email,))
            c.execute("""INSERT INTO ptb_web.email_challenges(token_hash,purpose,email,user_id,expires_at)
                VALUES(%s,'reset',%s,%s,clock_timestamp()+interval '15 minutes')""", (token_hash,email,user['id']))
            return True

    def consume_email(self, token_hash, purpose, password_hash=None):
        with self.connection() as c:
            # Read the lock keys first, then acquire in the same order as issue.
            row = c.execute('SELECT * FROM ptb_web.email_challenges WHERE token_hash=%s AND purpose=%s', (token_hash,purpose)).fetchone()
            if not row:
                return False
            self._lock(c, 'email:'+row['email'], *(['username:'+row['username']] if purpose=='register' else []))
            row = c.execute('''DELETE FROM ptb_web.email_challenges WHERE token_hash=%s AND purpose=%s
                AND expires_at>clock_timestamp() RETURNING *''', (token_hash,purpose)).fetchone()
            if not row:
                return False
            if purpose == 'register':
                if c.execute('SELECT 1 FROM ptb_web.emails WHERE email=%s', (row['email'],)).fetchone():
                    return False
                user = c.execute('''INSERT INTO ptb_accounts.users(id,username,password_hash) VALUES(%s,%s,%s)
                    ON CONFLICT(username) DO NOTHING RETURNING id''', (uuid4(),row['username'],row['password_hash'])).fetchone()
                if not user:
                    return False
                c.execute('INSERT INTO ptb_web.emails(user_id,email) VALUES(%s,%s)', (user['id'],row['email']))
            else:
                if not password_hash:
                    return False
                user = c.execute('UPDATE ptb_accounts.users SET password_hash=%s WHERE id=%s AND active RETURNING id',
                    (password_hash,row['user_id'])).fetchone()
                if not user:
                    return False
                c.execute('UPDATE ptb_accounts.sessions SET revoked_at=clock_timestamp() WHERE user_id=%s AND revoked_at IS NULL', (user['id'],))
                c.execute("DELETE FROM ptb_web.email_challenges WHERE user_id=%s AND purpose='reset'", (user['id'],))
            return True
