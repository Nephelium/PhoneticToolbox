-- M01-F REVIEW ONLY: explicit authorization required, additive metadata tables.
-- No existing row, P06/P07 schema, or ZIP limit is changed.
BEGIN;
CREATE TABLE ptb_jobs.acoustic_batch_version(version integer PRIMARY KEY CHECK(version=1));
CREATE TABLE ptb_jobs.acoustic_batches (
 id text PRIMARY KEY,
 owner_id uuid NOT NULL, project_id uuid NOT NULL,
 idempotency_key text NOT NULL, request_hash char(64) NOT NULL,
 operation text NOT NULL CHECK(operation IN ('acoustic_analysis','textgrid_segment')),
 config_snapshot text NOT NULL CHECK(octet_length(config_snapshot)<=16384),
 total integer NOT NULL CHECK(total BETWEEN 1 AND 1000),
 cancel_requested boolean NOT NULL DEFAULT FALSE,
 created_at double precision NOT NULL, updated_at double precision NOT NULL,
 closed_at double precision,
 UNIQUE(id,owner_id,project_id), UNIQUE(owner_id,idempotency_key),
 FOREIGN KEY(project_id,owner_id) REFERENCES ptb_accounts.projects(id,owner_id),
 CHECK(updated_at>=created_at), CHECK(closed_at IS NULL OR closed_at>=created_at)
);
CREATE TABLE ptb_jobs.acoustic_batch_items (
 batch_id text NOT NULL, ordinal integer NOT NULL CHECK(ordinal BETWEEN 0 AND 999),
 owner_id uuid NOT NULL, project_id uuid NOT NULL, audio_asset_id uuid NOT NULL,
 input_snapshot text NOT NULL CHECK(octet_length(input_snapshot)<=8192),
 child_job_id text, attempt integer NOT NULL DEFAULT 0 CHECK(attempt BETWEEN 0 AND 99),
 PRIMARY KEY(batch_id,ordinal), UNIQUE(batch_id,audio_asset_id), UNIQUE(child_job_id),
 FOREIGN KEY(batch_id,owner_id,project_id) REFERENCES ptb_jobs.acoustic_batches(id,owner_id,project_id),
 FOREIGN KEY(child_job_id,owner_id,project_id) REFERENCES ptb_jobs.jobs(id,owner_id,project_id),
 FOREIGN KEY(audio_asset_id,owner_id,project_id) REFERENCES ptb_storage.assets(id,owner_id,project_id)
);
CREATE INDEX acoustic_batches_owner ON ptb_jobs.acoustic_batches(owner_id,project_id,created_at,id);
CREATE INDEX acoustic_batches_pending ON ptb_jobs.acoustic_batches(created_at,id) WHERE closed_at IS NULL;
CREATE INDEX acoustic_batch_items_pending ON ptb_jobs.acoustic_batch_items(batch_id,ordinal) WHERE child_job_id IS NULL;
INSERT INTO ptb_jobs.acoustic_batch_version VALUES(1);
COMMIT;
