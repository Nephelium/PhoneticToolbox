"""P07-POLICY read-only aggregate preflight. No apply command or file cleanup.

Read one JSON line containing dsn from stdin; never print connection details.
Use only after authorization to inspect the intended database. Unit/integration
tests pass a connection to their own newly created synthetic cluster.
"""
import json
import sys
import hashlib
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from ptb_api.storage_policy import QUOTA_BYTES, LEGACY_QUOTA_BYTES


def metadata_fingerprints(conn):
    """Comparison digests only, never a backup or a copy of account content."""
    queries = {
        'assets': 'SELECT id,owner_id,project_id,name,kind,state,expected_bytes,size_bytes,reserved_bytes,sha256,created_at,expires_at FROM ptb_storage.assets ORDER BY id',
        'balances': 'SELECT owner_id,used_bytes,reserved_bytes FROM ptb_storage.quota_accounts ORDER BY owner_id',
        'jobs': 'SELECT id,snapshot,result_manifest,state,generation FROM ptb_jobs.jobs ORDER BY id',
    }
    result = {}
    for key, query in queries.items():
        digest = hashlib.sha256()
        for row in conn.execute(query):
            digest.update(json.dumps(row,sort_keys=True,default=str,ensure_ascii=False).encode('utf-8'))
            digest.update(b'\n')
        result[key] = digest.hexdigest()
    return result


def inspect(conn, storage_root=None):
    # Must be the first transaction command, including when called by tests.
    conn.execute('SET TRANSACTION READ ONLY')
    conn.execute("SET LOCAL statement_timeout='10s'")
    tables = [r['tablename'] for r in conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname='ptb_storage' ORDER BY tablename")]
    needed = {'state', 'quota_accounts', 'assets', 'job_assets', 'job_files_version'}
    if not needed.issubset(tables):
        return {'ready_for_review': False, 'reason': 'missing_storage_tables', 'tables': tables}
    state = conn.execute('SELECT * FROM ptb_storage.state').fetchone()
    constraints = {r['conname']: r['definition'] for r in conn.execute("""
        SELECT conname,pg_get_constraintdef(oid) AS definition FROM pg_constraint
        WHERE conrelid='ptb_storage.quota_accounts'::regclass ORDER BY conname""")}
    counts = dict(conn.execute('''SELECT count(*) AS accounts,
        count(*) FILTER(WHERE used_bytes+reserved_bytes>%s) AS over_quota_accounts,
        coalesce(sum(used_bytes),0) AS used_bytes, coalesce(sum(reserved_bytes),0) AS reserved_bytes,
        count(*) FILTER(WHERE quota_bytes<>%s) AS unexpected_old_quota
        FROM ptb_storage.quota_accounts''',(QUOTA_BYTES,LEGACY_QUOTA_BYTES)).fetchone())
    counts.update(conn.execute('''SELECT count(*) FILTER(WHERE state='uploading') AS in_flight_assets,
        count(*) FILTER(WHERE state='delete_failed') AS delete_failed_assets,
        count(*) FILTER(WHERE state!='deleted' AND expires_at<=extract(epoch FROM clock_timestamp())) AS expired_assets
        FROM ptb_storage.assets''').fetchone())
    counts['active_jobs'] = conn.execute("SELECT count(*) AS n FROM ptb_jobs.jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()['n']
    counts['ledger_mismatches'] = conn.execute('''SELECT count(*) AS n FROM ptb_storage.quota_accounts q
        LEFT JOIN (SELECT owner_id,sum(size_bytes) AS used,sum(reserved_bytes) AS reserved
            FROM ptb_storage.assets GROUP BY owner_id) a ON a.owner_id=q.owner_id
        WHERE q.used_bytes<>coalesce(a.used,0) OR q.reserved_bytes<>coalesce(a.reserved,0)''').fetchone()['n']
    version = state.get('policy_version', 1) if state else None
    versions = {}
    for table in ('ptb_accounts.schema_version', 'ptb_jobs.schema_version',
                  'ptb_storage.job_files_version', 'ptb_jobs.acoustic_batch_version'):
        exists = conn.execute('SELECT to_regclass(%s) AS name', (table,)).fetchone()['name']
        # Identifiers above are a fixed allowlist, never user input.
        versions[table] = [r['version'] for r in conn.execute('SELECT version FROM '+table+' ORDER BY version')] if exists else None
    disk = None
    if storage_root is not None:
        from ptb_api.storage import no_links, regular
        root = Path(storage_root).resolve()
        no_links(Path(storage_root).absolute())
        marker = root/'.ptb-storage.json'
        if not regular(marker):
            raise ValueError('invalid_storage_marker')
        instance = json.loads(marker.read_text('utf-8'))['instance_id']
        expected = {str(r['id'])+'.bin': r for r in conn.execute('SELECT id,state,size_bytes,reserved_bytes FROM ptb_storage.assets')}
        unknown = mismatched = 0
        for p in root.iterdir():
            if p.name in ('.ptb-storage.json','.ptb-storage.lock'):continue
            row = expected.get(p.name)
            if row is None or not regular(p):unknown += 1; continue
            size = p.stat().st_size
            if size != row['size_bytes']:mismatched += 1
        missing = sum(1 for name,r in expected.items() if r['state']!='deleted' and r['size_bytes']>0 and not (root/name).is_file())
        disk = dict(instance_matches=bool(state and str(state['instance_id'])==instance),
                    unknown_entries=unknown,size_mismatches=mismatched,missing_files=missing)
    ready = bool(state and state['version'] == 1 and version == 1
        and not counts['unexpected_old_quota'] and not counts['ledger_mismatches']
        and not state['frozen'] and all(v==[1] for v in versions.values())
        and (disk is None or (disk['instance_matches'] and not any(disk[k] for k in ('unknown_entries','size_mismatches','missing_files'))))
        and {'quota_accounts_check','quota_accounts_quota_bytes_check'}.issubset(constraints))
    return {'ready_for_review': ready, 'policy_version': version,
            'database': conn.execute('SELECT current_database() AS name').fetchone()['name'],
            'schema_versions': versions, 'storage_instance_id': str(state['instance_id']) if state else None,
            'storage_frozen': state['frozen'] if state else None, 'disk': disk,
            'metadata_fingerprints': metadata_fingerprints(conn),
            'counts': counts, 'constraints': constraints,
            'warning': 'Review is not authorization. Stop all writers; recheck under migration lock.'}


def main():
    try:
        config = json.loads(sys.stdin.readline())
        with psycopg.connect(config['dsn'], row_factory=dict_row, connect_timeout=5) as conn:
            report = inspect(conn, config.get('storage_root'))
        print(json.dumps(report, ensure_ascii=False, indent=2, default=int))
        return 0 if report['ready_for_review'] else 2
    except Exception:
        # Database exceptions can include identifiers and DSN/host information.
        print('P07 preflight failed; no migration was performed.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
