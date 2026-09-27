-- REVIEW ONLY. A release + explicit target migration approval required.
-- Run on the same database as ptb_jobs; no startup DDL. Preserve legacy rows.
BEGIN;
CREATE TABLE ptb_jobs.remote_nodes (
 id text PRIMARY KEY, token_hash text NOT NULL UNIQUE, token_until double precision NOT NULL,
 revoked integer NOT NULL DEFAULT 0, scopes text NOT NULL, operations text NOT NULL,
 runtime_hash text NOT NULL, slots integer NOT NULL CHECK(slots BETWEEN 1 AND 16),
 memory_bytes bigint NOT NULL CHECK(memory_bytes>0), last_seen double precision NOT NULL DEFAULT 0,
 healthy_count integer NOT NULL DEFAULT 0, cool_until double precision NOT NULL DEFAULT 0
);
CREATE TABLE ptb_jobs.remote_jobs (
 job_id text PRIMARY KEY REFERENCES ptb_jobs.jobs(id), runtime_hash text NOT NULL,
 memory_bytes bigint NOT NULL CHECK(memory_bytes>0), max_output_bytes bigint NOT NULL CHECK(max_output_bytes>0),
 max_attempts integer NOT NULL CHECK(max_attempts BETWEEN 1 AND 10),
 reason text NOT NULL DEFAULT 'waiting_node'
);
CREATE TABLE ptb_jobs.remote_attempts (
 id text PRIMARY KEY, job_id text NOT NULL REFERENCES ptb_jobs.jobs(id), generation integer NOT NULL,
 node_id text REFERENCES ptb_jobs.remote_nodes(id), location text NOT NULL,
 request_id text NOT NULL, worker_id text NOT NULL, runtime_hash text NOT NULL,
 parameter_hash text NOT NULL, font_hash text NOT NULL, input_hash text NOT NULL,
 phase text NOT NULL, started_at double precision NOT NULL, ended_at double precision,
 progress_at double precision NOT NULL, download_offset bigint NOT NULL DEFAULT 0,
 node_bytes bigint NOT NULL DEFAULT 0, error_code text, receipt text,
 UNIQUE(job_id,generation), UNIQUE(location,request_id)
);
CREATE UNIQUE INDEX remote_one_active ON ptb_jobs.remote_attempts(job_id) WHERE ended_at IS NULL;
CREATE TABLE ptb_jobs.remote_uploads (
 id text PRIMARY KEY, attempt_id text NOT NULL REFERENCES ptb_jobs.remote_attempts(id),
 output_key text NOT NULL, name text NOT NULL, size_bytes bigint NOT NULL,
 sha256 text NOT NULL, offset_bytes bigint NOT NULL DEFAULT 0,
 UNIQUE(attempt_id,output_key)
);
CREATE TABLE ptb_jobs.remote_chunks (
 upload_id text NOT NULL REFERENCES ptb_jobs.remote_uploads(id), offset_bytes bigint NOT NULL,
 size_bytes integer NOT NULL, sha256 text NOT NULL, PRIMARY KEY(upload_id,offset_bytes)
);
CREATE TABLE ptb_jobs.remote_reads (
 attempt_id text NOT NULL REFERENCES ptb_jobs.remote_attempts(id), asset_id text NOT NULL,
 offset_bytes bigint NOT NULL DEFAULT 0, PRIMARY KEY(attempt_id,asset_id)
);
COMMIT;
