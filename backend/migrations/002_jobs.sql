-- P06 review draft: explicit approval required; apply only to the dedicated P05 test database.
BEGIN;
CREATE SCHEMA ptb_jobs;
CREATE TABLE ptb_jobs.schema_version(version integer PRIMARY KEY);
CREATE TABLE ptb_jobs.jobs (
 id text PRIMARY KEY, owner_id uuid NOT NULL REFERENCES ptb_accounts.users(id),
 project_id uuid NOT NULL, idempotency_key text NOT NULL, request_hash char(64) NOT NULL,
 snapshot text NOT NULL CHECK(octet_length(snapshot)<=16384),
 state text NOT NULL CHECK(state IN ('queued','running','cancel_requested','cancelled','failed','interrupted','succeeded')),
 generation integer NOT NULL DEFAULT 0, worker_id text, lease_until double precision,
 deadline double precision NOT NULL, progress double precision NOT NULL DEFAULT 0 CHECK(progress>=0 AND progress<=1),
 result_manifest text, error_code text, event_seq integer NOT NULL DEFAULT 0,
 created_at double precision NOT NULL, updated_at double precision NOT NULL,
 UNIQUE(owner_id,idempotency_key), FOREIGN KEY(project_id,owner_id) REFERENCES ptb_accounts.projects(id,owner_id),
 CHECK((state='succeeded') = (result_manifest IS NOT NULL))
);
CREATE INDEX jobs_claim ON ptb_jobs.jobs(state,created_at,id);
CREATE INDEX jobs_owner ON ptb_jobs.jobs(owner_id,created_at,id);
CREATE TABLE ptb_jobs.events (
 job_id text NOT NULL REFERENCES ptb_jobs.jobs(id), sequence integer NOT NULL,
 state text NOT NULL, progress double precision NOT NULL, code text NOT NULL,
 created_at double precision NOT NULL, PRIMARY KEY(job_id,sequence)
);
INSERT INTO ptb_jobs.schema_version VALUES(1);
COMMIT;
