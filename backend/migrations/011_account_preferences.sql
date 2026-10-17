-- Account preferences are permanent account metadata, independent of workspaces.
-- Apply explicitly to the reviewed database; application startup never migrates.
BEGIN;
CREATE TABLE ptb_accounts.module_preferences (
    owner_id uuid NOT NULL REFERENCES ptb_accounts.users(id),
    module_id text NOT NULL CHECK (module_id IN ('APP','M01','M02','M03','M04','M05','M06','M07','M08','M09','M10','M11','M12','M13','M14','M15','M16','M17','M18')),
    revision bigint NOT NULL CHECK (revision > 0),
    settings jsonb NOT NULL CHECK (jsonb_typeof(settings) = 'object' AND octet_length(settings::text) <= 131072),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (owner_id, module_id)
);
COMMENT ON TABLE ptb_accounts.module_preferences IS 'Account settings only. No workspace/project/asset references or expiry; never removed by content cleanup.';
COMMIT;
