"""Opt-in, server-only leases. Closing a page never invokes an early-delete path."""
from uuid import UUID, uuid4
from contextlib import contextmanager
import psycopg
from psycopg.rows import dict_row
from fastapi import APIRouter, Depends, Request, Response
from pydantic import Field,PrivateAttr
from .models import WireModel, Identifier
from .account_store import hash_token
from .quota import StorageError

GRACE_SECONDS = 1800


class WorkspaceView(WireModel):
    _new: bool = PrivateAttr(default=False)
    id: str
    generation: int
    last_seen: float
    expires_at: float
    grace_seconds: int = GRACE_SECONDS


class WorkspaceOpen(WireModel):
    fresh: bool = False


class WorkspaceHeartbeat(WireModel):
    id: Identifier
    generation: int = Field(ge=1)


def view(row):
    return WorkspaceView(id=str(row['id']),generation=row['generation'],last_seen=row['last_seen'],expires_at=row['expires_at'])


class WorkspaceService:
    DEFAULT_QUOTA = 1_000_000_000
    UPLOAD_LIMIT = 100_000_000
    def __init__(self, storage):
        self.storage=storage

    def check_schema(self):
        with self.storage._locked() as c:
            if c.execute('SELECT version FROM ptb_web.workspace_policy').fetchall()!=[{'version':1}]:
                raise RuntimeError('Workspace migration required')
            if c.execute('SELECT version FROM ptb_web.capacity_policy').fetchall()!=[{'version':1}]:
                raise RuntimeError('Web capacity migration required')

    def default_quota(self, conn):
        return conn.execute('SELECT default_quota_bytes FROM ptb_web.capacity_policy WHERE version=1').fetchone()['default_quota_bytes']

    def expire_quota(self, conn):
        conn.execute('''UPDATE ptb_storage.quota_accounts SET quota_bytes=coalesce(web_base_quota_bytes,(SELECT default_quota_bytes FROM ptb_web.capacity_policy WHERE version=1)),
            web_override=(web_base_quota_bytes IS NOT NULL),web_override_expires_at=NULL WHERE web_override_expires_at<=clock_timestamp()''')

    def activate(self):
        self.check_schema()
        self.storage.workspaces=self

    @contextmanager
    def _read(self):
        # Identity queries never read a physical file. Taking the storage writer
        # lock here serialized every user's polling behind scientific output I/O.
        # Data reads and publication still recheck the lease in their own fence.
        with psycopg.connect(self.storage._dsn, autocommit=True, row_factory=dict_row,
                connect_timeout=5, options='-c statement_timeout=10000 -c default_transaction_read_only=on') as c:
            yield c

    @staticmethod
    def now(c):
        return c.execute('SELECT extract(epoch FROM clock_timestamp())::float8 AS t').fetchone()['t']

    def _active(self,c,owner,session_hash):
        row=c.execute("SELECT * FROM ptb_web.workspaces WHERE owner_id=%s AND session_hash=%s AND state='active'",(owner,session_hash)).fetchone()
        if not row or row['expires_at']<=self.now(c):
            raise StorageError('workspace_expired',410)
        return row

    def current(self,owner,session_hash):
        with self._read() as c:
            return self._active(c,owner,session_hash)

    def open(self,owner,session_hash,fresh=False):
        with self.storage._locked() as c, c.transaction():
            rows=c.execute('SELECT * FROM ptb_web.workspaces WHERE owner_id=%s AND session_hash=%s ORDER BY created_at DESC',(owner,session_hash)).fetchall()
            now=self.now(c)
            if rows and rows[0]['state']=='active' and rows[0]['expires_at']>now:
                return view(rows[0])
            if rows and not fresh:
                raise StorageError('workspace_expired',410)
            for row in rows:
                if row['state']=='active':self._close(c,row)
            active=c.execute('''SELECT 1 FROM ptb_accounts.sessions WHERE token_hash=%s AND user_id=%s
                AND revoked_at IS NULL AND expires_at>clock_timestamp() FOR SHARE''',(session_hash,owner)).fetchone()
            if not active:raise StorageError('authentication_required',401)
            row=c.execute('''INSERT INTO ptb_web.workspaces(id,owner_id,session_hash,last_seen,expires_at,created_at)
                VALUES(%s,%s,%s,%s,%s,%s) RETURNING *''',(uuid4(),owner,session_hash,now,now+GRACE_SECONDS,now)).fetchone()
            result=view(row);result._new=True
            return result

    def heartbeat(self,owner,session_hash,body):
        with self.storage._locked() as c, c.transaction():
            row=self._active(c,owner,session_hash)
            if str(row['id'])!=str(body.id) or row['generation']!=body.generation:
                raise StorageError('workspace_changed',409)
            now=self.now(c);deadline=now+GRACE_SECONDS
            row=c.execute('UPDATE ptb_web.workspaces SET last_seen=%s,expires_at=%s WHERE id=%s RETURNING *',(now,deadline,row['id'])).fetchone()
            c.execute("""UPDATE ptb_storage.assets a SET expires_at=%s FROM ptb_web.workspace_projects p
                WHERE a.project_id=p.project_id AND p.workspace_id=%s AND a.state IN ('uploading','ready')""",(deadline,row['id']))
            c.execute("""UPDATE ptb_storage.job_assets a SET input_expires_at=%s FROM ptb_web.workspace_projects p
                WHERE a.project_id=p.project_id AND p.workspace_id=%s AND a.role='input'""",(deadline,row['id']))
            return view(row)

    def project_deadline(self,c,owner,project):
        row=c.execute('''SELECT w.* FROM ptb_web.workspaces w JOIN ptb_web.workspace_projects p ON p.workspace_id=w.id
            WHERE p.owner_id=%s AND p.project_id=%s''',(owner,project)).fetchone()
        if not row or row['state']!='active' or row['expires_at']<=self.now(c):
            raise StorageError('workspace_expired',410)
        return row['expires_at']

    def authorize(self,owner,session_hash,*,project=None,asset=None,job=None,batch=None):
        with self._read() as c:
            row=self._active(c,owner,session_hash)
            # Fixed queries only. The authenticated session chooses the workspace.
            for value,query in [(asset,'SELECT project_id FROM ptb_storage.assets WHERE id=%s AND owner_id=%s'),
                (job,'SELECT project_id FROM ptb_jobs.jobs WHERE id=%s AND owner_id=%s'),
                (batch,'SELECT project_id FROM ptb_jobs.acoustic_batches WHERE id=%s AND owner_id=%s')]:
                if value is not None:
                    found=c.execute(query,(value,owner)).fetchone()
                    if not found:raise StorageError('resource_not_found',404)
                    project=found['project_id']
            if project is not None and not c.execute('SELECT 1 FROM ptb_web.workspace_projects WHERE workspace_id=%s AND project_id=%s AND owner_id=%s',(row['id'],project,owner)).fetchone():
                raise StorageError('resource_not_found',404)
            return row

    def projects(self,owner,session_hash,name=None,*,ensure_default=False):
        with self.storage._locked() as c, c.transaction():
            row=self._active(c,owner,session_hash)
            if ensure_default:
                existing=c.execute('''SELECT p.id,p.name,p.created_at FROM ptb_accounts.projects p
                    JOIN ptb_web.workspace_projects w ON w.project_id=p.id WHERE w.workspace_id=%s
                    ORDER BY p.created_at,p.id LIMIT 1''',(row['id'],)).fetchone()
                if existing:return dict(existing,id=str(existing['id']))
                name='默认项目'
            if name is not None:
                if c.execute('SELECT count(*) AS n FROM ptb_web.workspace_projects WHERE workspace_id=%s',(row['id'],)).fetchone()['n']>=100:
                    raise StorageError('project_limit_reached',409)
                project=c.execute('INSERT INTO ptb_accounts.projects(id,owner_id,name) VALUES(%s,%s,%s) RETURNING id,name,created_at',(uuid4(),owner,name)).fetchone()
                c.execute('INSERT INTO ptb_web.workspace_projects(workspace_id,owner_id,project_id) VALUES(%s,%s,%s)',(row['id'],owner,project['id']))
                return dict(project,id=str(project['id']))
            return [dict(p,id=str(p['id'])) for p in c.execute('''SELECT p.id,p.name,p.created_at FROM ptb_accounts.projects p
                JOIN ptb_web.workspace_projects w ON w.project_id=p.id WHERE w.workspace_id=%s ORDER BY p.created_at DESC,p.id DESC''',(row['id'],)).fetchall()]

    def _close(self,c,row):
        c.execute('SELECT pg_advisory_xact_lock(577606)')
        now=self.now(c)
        c.execute("UPDATE ptb_web.workspaces SET state='closed',generation=generation+1,closed_at=%s WHERE id=%s AND state='active'",(now,row['id']))
        c.execute('''UPDATE ptb_jobs.acoustic_batches b SET cancel_requested=true,closed_at=%s,updated_at=%s
            FROM ptb_web.workspace_projects p WHERE b.project_id=p.project_id AND p.workspace_id=%s AND b.closed_at IS NULL''',(now,now,row['id']))
        c.execute("""UPDATE ptb_jobs.jobs j SET state='cancelled',generation=generation+1,worker_id=NULL,lease_until=NULL,
            result_manifest=NULL,error_code='workspace_expired' FROM ptb_web.workspace_projects p
            WHERE j.project_id=p.project_id AND p.workspace_id=%s AND j.state IN ('queued','running','cancel_requested')""",(row['id'],))
        c.execute("""UPDATE ptb_storage.assets a SET state='deleting',expires_at=LEAST(a.expires_at,%s)
            FROM ptb_web.workspace_projects p WHERE a.project_id=p.project_id AND p.workspace_id=%s AND a.state!='deleted'""",(now,row['id']))

    def close(self,owner,session_hash):
        with self.storage._locked() as c, c.transaction():
            for row in c.execute("SELECT * FROM ptb_web.workspaces WHERE owner_id=%s AND session_hash=%s AND state='active'",(owner,session_hash)).fetchall():
                self._close(c,row)

    def release_expired_preview(self,preview):
        owner,token=preview.owner,preview.token
        if owner is None or token is None:return
        try:self.storage.metadata(owner[0],owner[1])
        except StorageError as exc:
            if exc.code in {'workspace_expired','asset_expired','asset_not_found','asset_unavailable'}:
                # release rechecks the exact identity under the preview lock.
                preview.release(owner,token)

    def cleanup(self):
        with self.storage._locked() as c:
            with c.transaction():
                for row in c.execute("""SELECT w.* FROM ptb_web.workspaces w JOIN ptb_accounts.sessions s ON s.token_hash=w.session_hash
                    WHERE w.state='active' AND (w.expires_at<=%s OR s.revoked_at IS NOT NULL OR s.expires_at<=clock_timestamp())
                    ORDER BY w.expires_at LIMIT 100""",(self.now(c),)).fetchall():
                    self._close(c,row)
            for row in c.execute("SELECT * FROM ptb_web.workspaces WHERE state='closed' ORDER BY closed_at LIMIT 100").fetchall():
                assets=c.execute("""SELECT a.* FROM ptb_storage.assets a JOIN ptb_web.workspace_projects p ON p.project_id=a.project_id
                    WHERE p.workspace_id=%s AND a.state!='deleted' LIMIT 100""",(row['id'],)).fetchall()
                for asset in assets:self.storage._delete(c,asset)
                if c.execute("""SELECT 1 FROM ptb_storage.assets a JOIN ptb_web.workspace_projects p ON p.project_id=a.project_id
                    WHERE p.workspace_id=%s AND a.state!='deleted' LIMIT 1""",(row['id'],)).fetchone():continue
                with c.transaction():
                    c.execute('SELECT pg_advisory_xact_lock(577606)')
                    ids=[p['project_id'] for p in c.execute('SELECT project_id FROM ptb_web.workspace_projects WHERE workspace_id=%s',(row['id'],)).fetchall()]
                    if ids:
                        c.execute('DELETE FROM ptb_jobs.acoustic_batch_items WHERE project_id=ANY(%s)',(ids,))
                        c.execute('DELETE FROM ptb_jobs.acoustic_batches WHERE project_id=ANY(%s)',(ids,))
                        c.execute('DELETE FROM ptb_jobs.events WHERE job_id IN (SELECT id FROM ptb_jobs.jobs WHERE project_id=ANY(%s))',(ids,))
                        c.execute('DELETE FROM ptb_storage.job_assets WHERE project_id=ANY(%s)',(ids,))
                        c.execute('DELETE FROM ptb_jobs.jobs WHERE project_id=ANY(%s)',(ids,))
                        c.execute('DELETE FROM ptb_storage.assets WHERE project_id=ANY(%s)',(ids,))
                        c.execute('DELETE FROM ptb_web.workspace_projects WHERE workspace_id=%s',(row['id'],))
                        c.execute('DELETE FROM ptb_accounts.projects WHERE id=ANY(%s)',(ids,))
                    c.execute("UPDATE ptb_web.workspaces SET state='cleaned' WHERE id=%s",(row['id'],))
            # Keep a content-free tombstone only while its session could reconnect.
            c.execute("""DELETE FROM ptb_web.workspaces w USING ptb_accounts.sessions s WHERE w.session_hash=s.token_hash
                AND w.state='cleaned' AND (s.revoked_at IS NOT NULL OR s.expires_at<=clock_timestamp())""")
            c.execute('DELETE FROM ptb_web.email_challenges WHERE expires_at<=clock_timestamp()')
            c.execute("DELETE FROM ptb_web.rate_limits WHERE window_start<clock_timestamp()-interval '1 day'")


def create_workspace_router(ctx, service, job_store=None):
    from .web_status import PublicLoad, ServerLoadView
    from .account_models import ProjectView
    load=PublicLoad(service,job_store)
    router=APIRouter(prefix='/api/v1/workspaces',tags=['workspaces'])
    def available():
        ctx.available()
        if service is None:raise StorageError('workspace_service_unavailable',503)
    def key(request):return hash_token(request.cookies.get(ctx.settings.cookie,''))
    @router.post('/open',response_model=WorkspaceView,operation_id='open_web_workspace',dependencies=[Depends(available)])
    def open_workspace(body:WorkspaceOpen,request:Request,response:Response,session=Depends(ctx.mutation)):
        value=service.open(session['id'],key(request),body.fresh)
        response.headers['X-PTB-Web-Use']='1' if value._new else '0'
        return value
    @router.post('/heartbeat',response_model=WorkspaceView,operation_id='heartbeat_web_workspace',dependencies=[Depends(available)])
    def heartbeat(body:WorkspaceHeartbeat,request:Request,session=Depends(ctx.mutation)):
        return service.heartbeat(session['id'],key(request),body)
    @router.post('/default-project',response_model=ProjectView,operation_id='open_default_web_project',dependencies=[Depends(available)])
    def default_project(request:Request,session=Depends(ctx.mutation)):
        return service.projects(session['id'],key(request),ensure_default=True)
    from .web_capacity import CapacityRequest, CapacityRequestView, PresenceView, request_capacity, latest_request, presence
    @router.get('/presence',response_model=PresenceView,operation_id='get_web_presence',dependencies=[Depends(available)])
    def online(include_load:bool=True,session=Depends(ctx.session)):
        result=presence(service)
        if include_load:result.server=load.sample()
        return result
    @router.get('/server-load',response_model=ServerLoadView,operation_id='get_web_server_load',dependencies=[Depends(available)])
    def server_load(details:bool=False,session=Depends(ctx.session)):
        result=load.sample(details=details)
        if result is None:raise StorageError('task_service_unavailable',503)
        return result
    @router.get('/capacity-request',response_model=CapacityRequestView|None,operation_id='get_web_capacity_request',dependencies=[Depends(available)])
    def capacity_status(session=Depends(ctx.session)):
        return latest_request(service,session['id'])
    @router.post('/capacity-request',response_model=CapacityRequestView,operation_id='request_web_capacity',dependencies=[Depends(available)])
    def capacity(body:CapacityRequest,session=Depends(ctx.mutation)):
        return request_capacity(service,session['id'],body)
    return router
