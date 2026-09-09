-- P07 REVIEW ONLY: explicit separate authorization, dedicated P05 test database.
BEGIN;
CREATE SCHEMA ptb_storage;
CREATE TABLE ptb_storage.state (
 singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton),
 version integer NOT NULL CHECK(version=1), instance_id uuid NOT NULL,
 frozen boolean NOT NULL DEFAULT true, reason text
);
-- instance_id is inserted by the approved initialization tool and matches the disk marker.
CREATE TABLE ptb_storage.quota_accounts (
 owner_id uuid PRIMARY KEY REFERENCES ptb_accounts.users(id),
 quota_bytes bigint NOT NULL DEFAULT 5000000000 CHECK(quota_bytes=5000000000),
 used_bytes bigint NOT NULL DEFAULT 0 CHECK(used_bytes>=0),
 reserved_bytes bigint NOT NULL DEFAULT 0 CHECK(reserved_bytes>=0),
 CHECK(used_bytes+reserved_bytes<=quota_bytes)
);
CREATE TABLE ptb_storage.assets (
 id uuid PRIMARY KEY, owner_id uuid NOT NULL REFERENCES ptb_storage.quota_accounts(owner_id),
 project_id uuid NOT NULL, name varchar(180) NOT NULL,
 kind text NOT NULL DEFAULT 'input' CHECK(kind='input'),
 state text NOT NULL CHECK(state IN ('uploading','ready','deleting','delete_failed','deleted')),
 idempotency_key varchar(128) NOT NULL, request_hash char(64) NOT NULL,
 size_bytes bigint NOT NULL DEFAULT 0 CHECK(size_bytes>=0),
 reserved_bytes bigint NOT NULL DEFAULT 0 CHECK(reserved_bytes>=0),
 expected_bytes bigint CHECK(expected_bytes BETWEEN 0 AND 5000000000),
 sha256 char(64), created_at double precision NOT NULL, expires_at double precision NOT NULL,
 error_code text, delete_attempts integer NOT NULL DEFAULT 0,
 last_delete_at double precision, deleted_at double precision,
 UNIQUE(owner_id,idempotency_key), FOREIGN KEY(project_id,owner_id) REFERENCES ptb_accounts.projects(id,owner_id),
 CHECK(state!='ready' OR (sha256 IS NOT NULL AND reserved_bytes=0)),
 CHECK(state!='deleted' OR (size_bytes=0 AND reserved_bytes=0))
);
CREATE INDEX assets_owner_project ON ptb_storage.assets(owner_id,project_id,created_at,id);
CREATE INDEX assets_expiry ON ptb_storage.assets(state,expires_at);
COMMIT;
