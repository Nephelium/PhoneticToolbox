-- P07 joint gate REVIEW ONLY. Not applied by the 003 storage validation command.
-- Requires explicit authorization after docs/testing/p07-job-assets-review.md.
-- Dedicated local test database only. Existing rows are not rewritten or deleted.
BEGIN;
-- Existing primary keys already guarantee uniqueness; composite keys additionally
-- enforce the same owner and project on both sides of every new file reference.
ALTER TABLE ptb_jobs.jobs ADD CONSTRAINT jobs_id_owner_project_key
 UNIQUE(id,owner_id,project_id);
ALTER TABLE ptb_storage.assets ADD CONSTRAINT assets_id_owner_project_key
 UNIQUE(id,owner_id,project_id);

-- Broaden only the resource kind. Existing input rows remain valid and unchanged.
ALTER TABLE ptb_storage.assets DROP CONSTRAINT assets_kind_check;
ALTER TABLE ptb_storage.assets ADD CONSTRAINT assets_kind_check
 CHECK(kind IN ('input','result','archive','temporary'));

CREATE TABLE ptb_storage.job_assets (
 job_id text NOT NULL, asset_id uuid NOT NULL,
 owner_id uuid NOT NULL, project_id uuid NOT NULL,
 role text NOT NULL CHECK(role IN ('input','output')),
 generation integer NOT NULL,
 input_sha256 char(64), input_expires_at double precision,
 created_at double precision NOT NULL,
 PRIMARY KEY(job_id,asset_id),
 FOREIGN KEY(job_id,owner_id,project_id)
  REFERENCES ptb_jobs.jobs(id,owner_id,project_id),
 FOREIGN KEY(asset_id,owner_id,project_id)
  REFERENCES ptb_storage.assets(id,owner_id,project_id),
 CHECK((role='input' AND generation=0 AND input_sha256 IS NOT NULL AND input_expires_at IS NOT NULL)
    OR (role='output' AND generation>0 AND input_sha256 IS NULL AND input_expires_at IS NULL))
);
CREATE UNIQUE INDEX job_assets_single_producer ON ptb_storage.job_assets(asset_id) WHERE role='output';
CREATE INDEX job_assets_input_expiry ON ptb_storage.job_assets(input_expires_at,job_id) WHERE role='input';
CREATE INDEX job_assets_asset_lookup ON ptb_storage.job_assets(asset_id,role);
CREATE TABLE ptb_storage.job_files_version(version integer PRIMARY KEY CHECK(version=1));
INSERT INTO ptb_storage.job_files_version(version) VALUES(1);
COMMIT;
