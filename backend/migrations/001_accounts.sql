-- P05 REVIEW DRAFT. Execute only after approval against the dedicated empty test DB.
-- One transaction; no changes to old-site tables and no DROP statements.
BEGIN;
CREATE SCHEMA ptb_accounts;
CREATE TABLE ptb_accounts.schema_version (version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE ptb_accounts.users (
 id uuid PRIMARY KEY, username varchar(64) NOT NULL UNIQUE,
 password_hash text NOT NULL, active boolean NOT NULL DEFAULT true,
 created_at timestamptz NOT NULL DEFAULT now(),
 CHECK (username ~ '^[a-z0-9][a-z0-9_.-]{2,63}$')
);
CREATE TABLE ptb_accounts.sessions (
 token_hash char(64) PRIMARY KEY, user_id uuid NOT NULL REFERENCES ptb_accounts.users(id),
 csrf_token varchar(128) NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 expires_at timestamptz NOT NULL, revoked_at timestamptz,
 CHECK (expires_at > created_at)
);
CREATE INDEX sessions_user ON ptb_accounts.sessions(user_id);
CREATE INDEX sessions_expiry ON ptb_accounts.sessions(expires_at);
CREATE TABLE ptb_accounts.projects (
 id uuid PRIMARY KEY, owner_id uuid NOT NULL REFERENCES ptb_accounts.users(id),
 name varchar(120) NOT NULL CHECK (length(btrim(name)) > 0),
 created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(id, owner_id)
);
CREATE INDEX projects_owner ON ptb_accounts.projects(owner_id, created_at, id);
CREATE TABLE ptb_accounts.login_attempts (
 bucket_key char(64) PRIMARY KEY, attempts integer NOT NULL CHECK (attempts > 0),
 window_start timestamptz NOT NULL
);
INSERT INTO ptb_accounts.schema_version(version) VALUES (1);
COMMIT;
