-- Additive opt-in workspace policy. No existing project/data is adopted or erased.
-- Requires the reviewed 008 identity schema and existing P05/P06/P07 tables.
BEGIN;
CREATE TABLE ptb_web.workspaces (
 id uuid PRIMARY KEY, owner_id uuid NOT NULL REFERENCES ptb_accounts.users(id),
 session_hash char(64) NOT NULL REFERENCES ptb_accounts.sessions(token_hash),
 generation integer NOT NULL DEFAULT 1 CHECK(generation>0),
 state text NOT NULL DEFAULT 'active' CHECK(state IN ('active','closed','cleaned')),
 last_seen double precision NOT NULL, expires_at double precision NOT NULL,
 created_at double precision NOT NULL, closed_at double precision,
 UNIQUE(id,owner_id), CHECK(expires_at=last_seen+1800)
);
CREATE UNIQUE INDEX workspaces_session_active ON ptb_web.workspaces(session_hash) WHERE state='active';
CREATE INDEX workspaces_due ON ptb_web.workspaces(state,expires_at);
CREATE TABLE ptb_web.workspace_projects (
 workspace_id uuid NOT NULL, owner_id uuid NOT NULL, project_id uuid PRIMARY KEY,
 FOREIGN KEY(workspace_id,owner_id) REFERENCES ptb_web.workspaces(id,owner_id),
 FOREIGN KEY(project_id,owner_id) REFERENCES ptb_accounts.projects(id,owner_id)
);
CREATE INDEX workspace_projects_workspace ON ptb_web.workspace_projects(workspace_id,project_id);
CREATE TABLE ptb_web.workspace_policy(version integer PRIMARY KEY CHECK(version=1));
INSERT INTO ptb_web.workspace_policy VALUES(1);

-- Fence publication inside the writer's transaction, including an old worker.
-- Lock order matches the storage/queue path. Legacy unbound projects are unchanged.
CREATE FUNCTION ptb_web.guard_project_write() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE w ptb_web.workspaces%ROWTYPE; project uuid;
BEGIN
 IF TG_TABLE_NAME='projects' THEN project := NEW.id;
 ELSE project := NEW.project_id; END IF;
 SELECT ws.* INTO w FROM ptb_web.workspaces ws
 JOIN ptb_web.workspace_projects p ON p.workspace_id=ws.id
 WHERE p.project_id=project FOR SHARE OF ws;
 IF FOUND AND (w.state!='active' OR w.expires_at<=extract(epoch FROM clock_timestamp())) THEN
   IF TG_TABLE_NAME='assets' THEN
     IF NEW.state IN ('deleting','delete_failed','deleted') THEN RETURN NEW; END IF;
   ELSIF TG_TABLE_NAME='jobs' THEN
     IF NEW.state IN ('cancel_requested','cancelled','failed','interrupted') THEN RETURN NEW; END IF;
   ELSIF TG_TABLE_NAME='acoustic_batches' THEN
     IF NEW.cancel_requested AND NEW.closed_at IS NOT NULL THEN RETURN NEW; END IF;
   END IF;
   RAISE EXCEPTION USING ERRCODE='P0001', MESSAGE='workspace_expired';
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER web_asset_fence BEFORE INSERT OR UPDATE ON ptb_storage.assets
 FOR EACH ROW EXECUTE FUNCTION ptb_web.guard_project_write();
CREATE TRIGGER web_job_fence BEFORE INSERT OR UPDATE ON ptb_jobs.jobs
 FOR EACH ROW EXECUTE FUNCTION ptb_web.guard_project_write();
CREATE TRIGGER web_project_fence BEFORE UPDATE ON ptb_accounts.projects
 FOR EACH ROW EXECUTE FUNCTION ptb_web.guard_project_write();
CREATE TRIGGER web_batch_fence BEFORE INSERT OR UPDATE ON ptb_jobs.acoustic_batches
 FOR EACH ROW EXECUTE FUNCTION ptb_web.guard_project_write();
COMMIT;
