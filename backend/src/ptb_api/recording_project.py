"""M16 project snapshots live in the existing quota/expiry asset store, no schema."""
import json
from .quota import StorageError
from .storage import public_asset

NAME='录音工程.m16.json'


def current(storage,owner,project):
    with storage._locked() as conn:
        storage.project_deadline(conn,owner,project)
        row=conn.execute("SELECT * FROM ptb_storage.assets WHERE owner_id=%s AND project_id=%s AND name=%s AND state='ready' ORDER BY created_at DESC,id DESC LIMIT 1",(owner,project,NAME)).fetchone()
        if row:storage._readable(conn,owner,row['id'])
        return public_asset(row) if row else None


def commit(storage,owner,project,asset_id,expected):
    with storage._locked() as conn:
        storage._writable(conn);storage.project_deadline(conn,owner,project)
        row=storage._readable(conn,owner,asset_id)
        if str(row['project_id'])!=project or row['size_bytes']>32_000_000:raise StorageError('recording_project_invalid',422)
        old=conn.execute("SELECT * FROM ptb_storage.assets WHERE owner_id=%s AND project_id=%s AND name=%s AND state='ready' ORDER BY created_at DESC,id DESC LIMIT 1",(owner,project,NAME)).fetchone()
        if old and str(old['id'])==str(asset_id):return public_asset(old)
        if not row['name'].endswith('.m16-draft.json'):raise StorageError('recording_project_invalid',422)
        if (str(old['id']) if old else None)!=expected:raise StorageError('recording_project_conflict',409)
        value=json.loads(storage._path(asset_id).read_bytes())
        if value.get('schema')!='m16-web-project/1' or not isinstance(value.get('project'),dict) or not isinstance(value.get('assets'),dict):raise StorageError('recording_project_invalid',422)
        project_data=value['project']
        if project_data.get('schema_version')!='ptb-recording/1' or len(project_data.get('takes',[]))>10000 or len(project_data.get('tasks',[]))>10000:raise StorageError('recording_project_invalid',422)
        references=set()
        for take in project_data.get('takes',[]):
            versions=take.get('versions',[])
            if not versions or len(versions)>10000 or not 0<=take.get('head',-1)<len(versions):raise StorageError('recording_project_invalid',422)
            for version in versions:
                ref=value['assets'].get(version['id'])
                if not isinstance(ref,dict):raise StorageError('recording_project_invalid',422)
                target=storage._readable(conn,owner,ref['id'])
                if str(target['project_id'])!=project or target['sha256']!=ref['sha256']:raise StorageError('input_unavailable',410)
                references.add(str(target['id']))
        # Publish the new immutable snapshot only after validating every reference.
        conn.execute('UPDATE ptb_storage.assets SET name=%s WHERE id=%s',(NAME,asset_id))
        result=public_asset(dict(row,name=NAME))
        # Superseded internal manifests contain no unique recording/version data.
        if old:storage._delete(conn,old)
        return result
