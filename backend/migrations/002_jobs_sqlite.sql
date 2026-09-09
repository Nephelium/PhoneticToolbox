-- P06 review draft: explicit initialization of an isolated local state database only.
BEGIN IMMEDIATE;
CREATE TABLE schema_version(version integer PRIMARY KEY);
CREATE TABLE jobs (
 id text PRIMARY KEY, owner_id text NOT NULL, project_id text NOT NULL,
 idempotency_key text NOT NULL, request_hash text NOT NULL,
 snapshot text NOT NULL CHECK(length(snapshot)<=16384),
 state text NOT NULL CHECK(state IN ('queued','running','cancel_requested','cancelled','failed','interrupted','succeeded')),
 generation integer NOT NULL DEFAULT 0, worker_id text, lease_until REAL,
 deadline REAL NOT NULL, progress REAL NOT NULL DEFAULT 0 CHECK(progress>=0 AND progress<=1),
 result_manifest text, error_code text, event_seq integer NOT NULL DEFAULT 0,
 created_at REAL NOT NULL, updated_at REAL NOT NULL,
 UNIQUE(owner_id,idempotency_key), CHECK((state='succeeded') = (result_manifest IS NOT NULL))
);
CREATE INDEX jobs_claim ON jobs(state,created_at,id);
CREATE INDEX jobs_owner ON jobs(owner_id,created_at,id);
CREATE TABLE events (
 job_id text NOT NULL REFERENCES jobs(id), sequence integer NOT NULL,
 state text NOT NULL, progress REAL NOT NULL, code text NOT NULL,
 created_at REAL NOT NULL, PRIMARY KEY(job_id,sequence)
);
INSERT INTO schema_version VALUES(1);
COMMIT;
