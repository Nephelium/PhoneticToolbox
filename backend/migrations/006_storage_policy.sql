-- P07-POLICY REVIEW CANDIDATE. Never auto-apply to an existing/service database.
-- Confirm docs/testing/p07-policy-migration-review.md and stop ALL writers first.
-- Existing asset expiry, manifests, job snapshots and reservations are preserved.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
SELECT pg_advisory_xact_lock(577707);
LOCK TABLE ptb_storage.state, ptb_storage.quota_accounts, ptb_storage.assets
 IN ACCESS EXCLUSIVE MODE;

DO $$
BEGIN
 IF (SELECT count(*) FROM ptb_storage.state WHERE version=1)<>1
    OR EXISTS(SELECT 1 FROM ptb_storage.quota_accounts WHERE quota_bytes<>5000000000)
    OR EXISTS(SELECT 1 FROM ptb_storage.quota_accounts q LEFT JOIN
      (SELECT owner_id,sum(size_bytes) AS used,sum(reserved_bytes) AS reserved
       FROM ptb_storage.assets GROUP BY owner_id) a ON a.owner_id=q.owner_id
      WHERE q.used_bytes<>coalesce(a.used,0) OR q.reserved_bytes<>coalesce(a.reserved,0)) THEN
  RAISE EXCEPTION 'policy_preflight_mismatch';
 END IF;
END;
$$;

ALTER TABLE ptb_storage.state ADD COLUMN policy_version integer NOT NULL DEFAULT 1
 CHECK(policy_version IN (1,2));
ALTER TABLE ptb_storage.state ADD COLUMN policy_switched_at double precision;
ALTER TABLE ptb_storage.assets ADD COLUMN policy_version integer NOT NULL DEFAULT 1
 CHECK(policy_version IN (1,2));
-- Existing rows stay version 1. New rows are explicitly governed by policy 2.
ALTER TABLE ptb_storage.assets ALTER COLUMN policy_version SET DEFAULT 2;

ALTER TABLE ptb_storage.quota_accounts DROP CONSTRAINT quota_accounts_quota_bytes_check;
ALTER TABLE ptb_storage.quota_accounts DROP CONSTRAINT quota_accounts_check;
ALTER TABLE ptb_storage.quota_accounts ALTER COLUMN quota_bytes SET DEFAULT 1000000000;
UPDATE ptb_storage.quota_accounts SET quota_bytes=1000000000;
ALTER TABLE ptb_storage.quota_accounts ADD CONSTRAINT quota_accounts_quota_bytes_check
 CHECK(quota_bytes=1000000000);

-- Grandfather the OLD balance, never grant new capacity above the quota.
-- Row locking serializes concurrent changes. Crash recovery may move reserved
-- bytes to used without changing the total; reductions and deletion always work.
CREATE FUNCTION ptb_storage.guard_policy_quota() RETURNS trigger
 LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.used_bytes+NEW.reserved_bytes>NEW.quota_bytes THEN
  IF TG_OP='INSERT' THEN
   RAISE EXCEPTION 'quota_exceeded' USING ERRCODE='23514';
  ELSIF NEW.used_bytes+NEW.reserved_bytes>OLD.used_bytes+OLD.reserved_bytes THEN
   RAISE EXCEPTION 'quota_exceeded' USING ERRCODE='23514';
  END IF;
 END IF;
 RETURN NEW;
END;
$$;
CREATE TRIGGER policy_quota_growth BEFORE INSERT OR UPDATE ON ptb_storage.quota_accounts
 FOR EACH ROW EXECUTE FUNCTION ptb_storage.guard_policy_quota();

-- Keep the historical expected_bytes CHECK (5 GB) for old rows. Only newly
-- created assets must satisfy the current ceiling; a legacy in-flight upload
-- can still release reservations/finalize its already written bytes.
CREATE FUNCTION ptb_storage.guard_new_asset_policy() RETURNS trigger
 LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.policy_version<>2 OR NEW.expected_bytes>1000000000 THEN
  RAISE EXCEPTION 'invalid_new_asset_policy' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END;
$$;
CREATE TRIGGER new_asset_policy BEFORE INSERT ON ptb_storage.assets
 FOR EACH ROW EXECUTE FUNCTION ptb_storage.guard_new_asset_policy();

UPDATE ptb_storage.state SET policy_version=2,
 policy_switched_at=extract(epoch FROM clock_timestamp());
ALTER TABLE ptb_storage.state ALTER COLUMN policy_version SET DEFAULT 2;
COMMIT;
