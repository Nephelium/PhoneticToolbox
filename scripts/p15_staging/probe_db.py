"""Read-only target DB/storage binding inspection; config is one stdin JSON line.

Requires authorization for the identified target. Never starts PostgreSQL,
applies schema, invokes Storage.recover/cleanup or prints credentials/filenames.
"""
import json
from pathlib import Path
import sys

from host import database_state


def inspect(config):
    import psycopg
    from psycopg.rows import dict_row
    from ptb_api.storage import no_links,regular
    root=Path(config['storage_root'])
    no_links(root)
    marker=root/'.ptb-storage.json'
    if not regular(marker) or not regular(root/'.ptb-storage.lock'):
        raise ValueError('storage_not_initialized')
    instance=json.loads(marker.read_text('utf-8'))['instance_id']
    if instance!=config['storage_instance_id']:raise ValueError('storage_instance_mismatch')
    report=database_state(config['dsn'],instance)
    with psycopg.connect(config['dsn'],row_factory=dict_row,connect_timeout=5) as conn:
        conn.execute('SET TRANSACTION READ ONLY')
        conn.execute("SET LOCAL statement_timeout='10s'")
        report['assets']=dict(conn.execute("""SELECT count(*) AS total,
            count(*) FILTER(WHERE policy_version=1) AS legacy,
            count(*) FILTER(WHERE policy_version=2) AS current,
            count(*) FILTER(WHERE state='delete_failed') AS delete_failed,
            coalesce(sum(size_bytes),0) AS used_bytes,
            coalesce(sum(reserved_bytes),0) AS reserved_bytes
            FROM ptb_storage.assets""").fetchone())
        report['jobs']=[dict(r) for r in conn.execute('SELECT operation,state,count(*) AS n FROM ptb_jobs.jobs GROUP BY operation,state')]
        report['ledger_mismatches']=conn.execute("""SELECT count(*) AS n FROM ptb_storage.quota_accounts q
            LEFT JOIN (SELECT owner_id,sum(size_bytes) used,sum(reserved_bytes) reserved
                FROM ptb_storage.assets GROUP BY owner_id) a ON a.owner_id=q.owner_id
            WHERE q.used_bytes<>coalesce(a.used,0) OR q.reserved_bytes<>coalesce(a.reserved,0)""").fetchone()['n']
    report.update(schema='p15-db-probe/1',instance_matches=True,read_only=True,
        actual_policy_boundary_tested=False,ready_for_deployment=False)
    return report


def main():
    try:
        c=json.loads(sys.stdin.readline(16385))
        print(json.dumps(inspect(c),ensure_ascii=False,default=int))
        return 0
    except Exception:
        print('{"read_only":true,"success":false,"detail":"target_policy_or_binding_check_failed"}')
        return 2


if __name__=='__main__':sys.exit(main())
