-- Reviewable additive migration. Application startup never runs this file.
-- Apply only to an explicitly approved target after 001_accounts.sql.
-- Existing users, sessions, projects and research data remain unchanged.
BEGIN;
CREATE SCHEMA ptb_web;
CREATE TABLE ptb_web.schema_version (
 version integer PRIMARY KEY CHECK(version=1), applied_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE ptb_web.emails (
 user_id uuid PRIMARY KEY REFERENCES ptb_accounts.users(id),
 email varchar(254) NOT NULL UNIQUE, verified_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE ptb_web.email_challenges (
 token_hash char(64) PRIMARY KEY,
 purpose text NOT NULL CHECK(purpose IN ('register','reset')),
 email varchar(254) NOT NULL, username varchar(64), password_hash text,
 user_id uuid REFERENCES ptb_accounts.users(id),
 expires_at timestamptz NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 CHECK((purpose='register' AND username IS NOT NULL AND password_hash IS NOT NULL AND user_id IS NULL)
    OR (purpose='reset' AND username IS NULL AND password_hash IS NULL AND user_id IS NOT NULL))
);
CREATE INDEX email_challenges_expiry ON ptb_web.email_challenges(expires_at);
CREATE UNIQUE INDEX email_challenges_address ON ptb_web.email_challenges(email,purpose);
CREATE TABLE ptb_web.rate_limits (
 bucket_key char(64) PRIMARY KEY, attempts integer NOT NULL CHECK(attempts>0),
 window_start timestamptz NOT NULL
);
INSERT INTO ptb_web.schema_version(version) VALUES(1);
COMMIT;
