"""Explicitly approved 006 runner for the dedicated local test database only.

No backup, production target, automatic recovery, cleanup or API restart.
The default/show command never opens a database. Approval flags are a guard,
not a substitute for the user's authorization of this exact target.
"""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = ROOT/'backend/migrations/006_storage_policy.sql'
REVIEWED_SHA256 = 'bf5eb0d18eb3d242db98f026c9a32c55a90bf57b21589efe36365a974866f247'
DATABASE = 'ptb_p05_test_20260909'
STORAGE_ROOT = ROOT/'output/validation/p07/storage'
INSTANCE = 'efe95c3e-708c-4c00-b9f3-4ec19f7d49b1'


def reviewed_sql(digest):
    raw = SQL_PATH.read_bytes()
    if digest != REVIEWED_SHA256 or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('reviewed_sql_hash_mismatch')
    return raw.decode('utf-8')


def apply_checked(conn, root, database, instance, sql):
    """Caller holds the storage OS/session lock and has stopped all writers."""
    from p07_policy_preflight import inspect, metadata_fingerprints
    with conn.transaction():
        before = inspect(conn, root)
        if before['database'] != database or before['storage_instance_id'] != instance:
            raise ValueError('target_identity_mismatch')
        if not before['ready_for_review']:
            raise ValueError('preflight_not_ready')
        # Queued jobs and incomplete uploads may remain. An owned worker must
        # finish first; never cancel/recover a user's in-flight task implicitly.
        active = conn.execute("SELECT count(*) AS n FROM ptb_jobs.jobs WHERE state IN ('running','cancel_requested')").fetchone()['n']
        if active:
            raise ValueError('active_workers_require_review')
    try:
        conn.execute(sql)
    except Exception:
        conn.execute('ROLLBACK')
        raise
    # SQL has committed. Failure here must leave services closed, not attempt a
    # destructive restore or pretend a ROLLBACK can undo the committed migration.
    with conn.transaction():
        conn.execute('SET TRANSACTION READ ONLY')
        after = metadata_fingerprints(conn)
        state = conn.execute('SELECT * FROM ptb_storage.state').fetchone()
        wrong_quota = conn.execute('SELECT count(*) AS n FROM ptb_storage.quota_accounts WHERE quota_bytes<>1000000000').fetchone()['n']
        wrong_legacy = conn.execute('SELECT count(*) AS n FROM ptb_storage.assets WHERE policy_version<>1').fetchone()['n']
        if after != before['metadata_fingerprints'] or state['policy_version'] != 2 or wrong_quota or wrong_legacy:
            raise ValueError('postcommit_verification_failed_keep_writes_closed')
    return dict(schema_applied=['006'], policy_version=2, preserved_fingerprints=after,
                writes_reopened=False, module_web_validation='pending')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('show','apply'), nargs='?', default='show')
    parser.add_argument('--approved-dedicated-p07-006', action='store_true')
    parser.add_argument('--reviewed-sha256')
    args = parser.parse_args()
    if args.command == 'show':
        print(json.dumps(dict(database=DATABASE, storage_root=str(STORAGE_ROOT), instance_id=INSTANCE,
                              sql_sha256=hashlib.sha256(SQL_PATH.read_bytes()).hexdigest(),
                              schema_applied=[], writes_reopened=False),indent=2))
        return 0
    if not args.approved_dedicated_p07_006:
        parser.error('Explicit target-specific DDL/process authorization required')
    sql = reviewed_sql(args.reviewed_sha256)
    from run_m01_validation import owned_postgres
    from ptb_api.storage import Storage
    out = ROOT/'output/validation/p07-closeout-20260927'/('migration-'+uuid4().hex)
    out.mkdir(parents=True)
    report = dict(database=DATABASE, storage_root=str(STORAGE_ROOT), sql_sha256=REVIEWED_SHA256,
                  writes_reopened=False, success=False)
    try:
        # This harness refuses an already-running cluster, validates its fixed
        # PGDATA/runtime, and stops only the exact instance it started itself.
        with owned_postgres(out) as config:
            storage = Storage(config['dsn'], STORAGE_ROOT)
            with storage._locked() as conn:
                if conn.execute('SELECT count(*) AS n FROM pg_stat_activity WHERE datname=current_database() AND pid<>pg_backend_pid()').fetchone()['n']:
                    raise ValueError('other_database_sessions_require_review')
                report.update(apply_checked(conn, STORAGE_ROOT, DATABASE, INSTANCE, sql))
        report['success'] = True
    except Exception as exc:
        # Never serialize database exceptions, DSNs or raw private config.
        report['error_type'] = type(exc).__name__
        report['action'] = 'keep_writes_closed_and_review_database_commit_state'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(out/'report.json')
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
