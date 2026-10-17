-- Explicit web-only migration. Apply only to an approved database, with writers stopped.
-- Existing data is retained, including over-quota rows; the growth fence stays active.
BEGIN;
SELECT pg_advisory_xact_lock(577707);
LOCK TABLE ptb_storage.quota_accounts IN ACCESS EXCLUSIVE MODE;
ALTER TABLE ptb_storage.quota_accounts DROP CONSTRAINT quota_accounts_quota_bytes_check;
ALTER TABLE ptb_storage.quota_accounts ADD CONSTRAINT quota_accounts_quota_bytes_check CHECK(quota_bytes>0);
ALTER TABLE ptb_storage.quota_accounts ALTER COLUMN quota_bytes SET DEFAULT 1000000000;
UPDATE ptb_storage.quota_accounts SET quota_bytes=1000000000;
ALTER TABLE ptb_storage.quota_accounts ADD COLUMN web_override boolean NOT NULL DEFAULT false;
ALTER TABLE ptb_storage.quota_accounts ADD COLUMN web_base_quota_bytes bigint CHECK(web_base_quota_bytes BETWEEN 100000000 AND 10000000000);
ALTER TABLE ptb_storage.quota_accounts ADD COLUMN web_override_expires_at timestamptz;
CREATE INDEX web_quota_expiry ON ptb_storage.quota_accounts(web_override_expires_at) WHERE web_override_expires_at IS NOT NULL;
CREATE TABLE ptb_web.capacity_policy(version integer PRIMARY KEY CHECK(version=1), default_quota_bytes bigint NOT NULL DEFAULT 1000000000 CHECK(default_quota_bytes BETWEEN 100000000 AND 10000000000));
INSERT INTO ptb_web.capacity_policy(version) VALUES(1);
CREATE TABLE ptb_web.capacity_requests(
 id uuid PRIMARY KEY,
 owner_id uuid NOT NULL REFERENCES ptb_accounts.users(id),
 requested_bytes bigint NOT NULL CHECK(requested_bytes>0 AND requested_bytes<=10000000000),
 reason text NOT NULL CHECK(length(reason) BETWEEN 1 AND 500),
 duration text NOT NULL DEFAULT 'today' CHECK(duration IN ('today','7d','30d')),
 state text NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','approved','declined')),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 decided_at timestamptz,
 granted_bytes bigint,
 granted_duration text CHECK(granted_duration IN ('today','7d','30d')),
 expires_at timestamptz,
 decision_note text
);
CREATE UNIQUE INDEX capacity_one_pending ON ptb_web.capacity_requests(owner_id) WHERE state='pending';
CREATE INDEX workspace_recent_accounts ON ptb_web.workspaces(last_seen,owner_id) WHERE state='active';
COMMIT;
