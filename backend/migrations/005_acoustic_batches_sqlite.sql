-- M01-F REVIEW ONLY: additive migration of the explicitly approved P06 test DB.
-- The composite index enables matching local owner/project foreign keys; no rewrite.
BEGIN IMMEDIATE;
CREATE UNIQUE INDEX m01_jobs_identity ON jobs(id,owner_id,project_id);
CREATE TABLE acoustic_batch_version(version integer PRIMARY KEY CHECK(version=1));
CREATE TABLE acoustic_batches (
 id text PRIMARY KEY, owner_id text NOT NULL CHECK(owner_id='local'),
 project_id text NOT NULL CHECK(project_id='00000000-0000-4000-8000-000000000001'),
 idempotency_key text NOT NULL, request_hash text NOT NULL CHECK(length(request_hash)=64),
 operation text NOT NULL CHECK(operation IN ('acoustic_analysis','textgrid_segment')),
 config_snapshot text NOT NULL CHECK(length(CAST(config_snapshot AS BLOB))<=16384),
 total integer NOT NULL CHECK(total BETWEEN 1 AND 1000),
 cancel_requested integer NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),
 created_at REAL NOT NULL, updated_at REAL NOT NULL, closed_at REAL,
 UNIQUE(id,owner_id,project_id), UNIQUE(owner_id,idempotency_key),
 CHECK(updated_at>=created_at), CHECK(closed_at IS NULL OR closed_at>=created_at)
);
CREATE TABLE acoustic_batch_items (
 batch_id text NOT NULL, ordinal integer NOT NULL CHECK(ordinal BETWEEN 0 AND 999),
 owner_id text NOT NULL, project_id text NOT NULL, audio_asset_id text NOT NULL,
 input_snapshot text NOT NULL CHECK(length(CAST(input_snapshot AS BLOB))<=8192),
 child_job_id text, attempt integer NOT NULL DEFAULT 0 CHECK(attempt BETWEEN 0 AND 99),
 PRIMARY KEY(batch_id,ordinal), UNIQUE(batch_id,audio_asset_id), UNIQUE(child_job_id),
 FOREIGN KEY(batch_id,owner_id,project_id) REFERENCES acoustic_batches(id,owner_id,project_id),
 FOREIGN KEY(child_job_id,owner_id,project_id) REFERENCES jobs(id,owner_id,project_id)
);
CREATE INDEX acoustic_batches_owner ON acoustic_batches(owner_id,project_id,created_at,id);
CREATE INDEX acoustic_batches_pending ON acoustic_batches(created_at,id) WHERE closed_at IS NULL;
CREATE INDEX acoustic_batch_items_pending ON acoustic_batch_items(batch_id,ordinal) WHERE child_job_id IS NULL;
INSERT INTO acoustic_batch_version VALUES(1);
COMMIT;
