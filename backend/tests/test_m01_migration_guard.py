"""Read-only migration review guards; does NOT execute 005 or claim DB acceptance."""
import hashlib
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('kind',['postgres','sqlite'])
def test_show_and_unapproved_apply_never_touch_database(kind):
    local=ROOT/'output/validation/p06/local-state.sqlite3'
    before=hashlib.sha256(local.read_bytes()).hexdigest() if local.exists() else None
    for options,expected in ((['show'],0),(['apply'],2),(['apply','--approved-m01-batch-schema','--reviewed-sha256','0'*64],2)):
        result=subprocess.run([sys.executable,'scripts/m01_database.py',*options,'--kind',kind],cwd=ROOT,input='',capture_output=True,text=True,timeout=10)
        assert result.returncode==expected
    assert (hashlib.sha256(local.read_bytes()).hexdigest() if local.exists() else None)==before


def test_proposed_sql_is_additive_and_keeps_operation_specific_zip_limit():
    for name in ('005_acoustic_batches.sql','005_acoustic_batches_sqlite.sql'):
        text=(ROOT/'backend/migrations'/name).read_text('utf-8')
        lines='\n'.join(x for x in text.splitlines() if not x.lstrip().startswith('--')).upper()
        assert 'DROP ' not in lines and 'DELETE ' not in lines and 'ALTER TABLE ' not in lines and 'UPDATE ' not in lines
        assert lines.count('CREATE TABLE ')==3
        assert 'FOREIGN KEY(CHILD_JOB_ID,OWNER_ID,PROJECT_ID)' in lines
