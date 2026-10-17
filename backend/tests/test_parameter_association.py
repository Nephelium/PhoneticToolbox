from contextlib import contextmanager
from types import SimpleNamespace
import json
from ptb_worker.acoustic_batches import AcousticBatches
from ptb_worker.acoustic_files import AcousticFiles
from ptb_worker.store import JobError

def test_table_association_uses_source_query_not_recent_job_list_and_skips_unreadable_output():
    calls=[]
    def row(files):return dict(snapshot=json.dumps({'operation':'acoustic_analysis','input_refs':{'audio':{'sha256':'source'}}}),result_manifest=json.dumps({'files':files}))
    rows=[row([{'id':'expired','name':'result.ptb.sqlite','sha256':'old'}]),row([{'id':'json','name':'result.ptb.json','sha256':'json'},{'id':'xlsx','name':'result.xlsx','sha256':'x'},{'id':'sqlite','name':'result.ptb.sqlite','sha256':'s'}])]
    class Transaction:
        def execute(self,sql,args):
            assert "snapshot::jsonb->'input_refs'->'audio'->>'sha256'=?" in sql
            assert args==('owner','project','source');return rows
        def now(self):return 1000
    class Resources:
        @contextmanager
        def batch_transaction(self):yield Transaction()
        def read_parameter_reference(self,tx,owner,project,ref):
            calls.append(ref['asset_id'])
            if ref['asset_id']=='expired':raise JobError('input_unavailable',409)
    batch=object.__new__(AcousticBatches);batch.jobs=SimpleNamespace(postgres=True);batch.resources=Resources()
    assert batch.parent_result('owner','project','source',table=True)=={'asset_id':'sqlite','name':'result.ptb.sqlite','sha256':'s'}
    assert calls==['expired','sqlite']

def test_display_table_remains_available_in_last_five_minutes_and_binds_owner_project_hash():
    seen=[]
    def readable(conn,owner,asset):seen.append((owner,asset));return dict(project_id='project',sha256='s',size_bytes=100,name='result.ptb.sqlite',expires_at=101)
    resources=object.__new__(AcousticFiles);resources.storage=SimpleNamespace(_readable=readable)
    resources.read_parameter_reference(SimpleNamespace(conn='connection'),'owner','project',{'asset_id':'table','sha256':'s'})
    assert seen==[('owner','table')]
    try:resources.read_parameter_reference(SimpleNamespace(conn='connection'),'owner','other-project',{'asset_id':'table','sha256':'s'})
    except JobError as exc:assert exc.code=='input_unavailable'
    else:raise AssertionError('cross-project association')
