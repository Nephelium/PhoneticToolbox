"""Read-only schema/policy gate on the existing dedicated test cluster; no DDL."""
import json
from uuid import uuid4
import psycopg
from psycopg.rows import dict_row
from run_m01_validation import owned_postgres,ROOT
from ptb_api.storage_policy import POLICY_VERSION,LEGACY_POLICY_VERSION


def main():
    out=ROOT/'output/validation/m08-wiring'/('pg-gate-'+uuid4().hex);out.mkdir(parents=True)
    report={'read_only':True,'schema_applied':[]}
    try:
        with owned_postgres(out) as config:
            with psycopg.connect(config['dsn'],row_factory=dict_row) as conn:
                conn.execute('SET TRANSACTION READ ONLY')
                state=conn.execute('SELECT * FROM ptb_storage.state').fetchone()
                actual=state.get('policy_version',LEGACY_POLICY_VERSION)
                report.update(actual_policy=actual,required_policy=POLICY_VERSION,write_gate_ready=actual==POLICY_VERSION)
    except RuntimeError as exc:report['blocked']=str(exc)
    finally:
        (out/'gate.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report));print(out)


if __name__=='__main__':main()
