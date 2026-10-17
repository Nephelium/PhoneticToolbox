"""Opt-in internal operations API. Public managers authenticate at feedback service."""
import hmac
import shutil
from uuid import UUID
from typing import Literal
from fastapi import APIRouter,Depends,HTTPException,Request,Response
from .web_capacity import public,presence,decide,set_quota
from .web_status import PublicLoad


from .web_ops_models import Decision,QuotaChange


class WebOperations:
    def __init__(self,workspaces,jobs):
        self.service=workspaces;self.load=PublicLoad(workspaces,jobs)

    def snapshot(self):
        s=self.service
        load=self.load.sample();online=presence(s)
        disk=shutil.disk_usage(s.storage.root)
        with s._read() as c:
            quota=c.execute('SELECT coalesce(sum(used_bytes),0) AS used,coalesce(sum(reserved_bytes),0) AS reserved FROM ptb_storage.quota_accounts').fetchone()
            cleanup=c.execute("SELECT count(*) FILTER(WHERE state IN ('deleting','delete_failed')) AS pending,count(*) FILTER(WHERE state='delete_failed') AS failed FROM ptb_storage.assets").fetchone()
            pending=c.execute("SELECT count(*) AS n FROM ptb_web.capacity_requests WHERE state='pending'").fetchone()['n']
            default=s.default_quota(c)
        return dict(measured_at=online.measured_at,online_accounts=online.online_accounts,
            load=load.model_dump() if load else None,disk_total_bytes=disk.total,disk_free_bytes=disk.free,
            disk_safety_bytes=s.storage.min_free_bytes,used_bytes=int(quota['used']),reserved_bytes=int(quota['reserved']),
            cleanup_pending=cleanup['pending'],cleanup_failed=cleanup['failed'],pending_requests=pending,default_quota_bytes=default)

    def requests(self,state='pending',before=None):
        # Bounded keyset pagination; research filenames/content never enter the view.
        with self.service._read() as c:
            rows=c.execute('''SELECT r.*,u.username,q.quota_bytes,q.used_bytes,q.reserved_bytes
                FROM ptb_web.capacity_requests r JOIN ptb_accounts.users u ON u.id=r.owner_id
                LEFT JOIN ptb_storage.quota_accounts q ON q.owner_id=r.owner_id
                WHERE r.state=%s AND (%s::uuid IS NULL OR (r.created_at,r.id)<
                    (SELECT created_at,id FROM ptb_web.capacity_requests WHERE id=%s))
                ORDER BY r.created_at DESC,r.id DESC LIMIT 51''',(state,before,before)).fetchall()
            default=self.service.default_quota(c)
        items=[]
        for r in rows[:50]:
            items.append(dict(**public(r).model_dump(mode='json'),owner_id=str(r['owner_id']),username=r['username'],reason=r['reason'],
                quota_bytes=r['quota_bytes'] or default,used_bytes=r['used_bytes'] or 0,reserved_bytes=r['reserved_bytes'] or 0))
        return dict(requests=items,next_cursor=str(rows[49]['id']) if len(rows)>50 else None)

    def decision(self,body):return decide(self.service,body.request_id,granted_bytes=body.granted_bytes,note=body.note,duration=body.duration).model_dump(mode='json')
    def quota(self,body):return set_quota(self.service,body.quota_bytes,body.owner_id)


def create_web_admin_router(service,key):
    if not isinstance(key,str) or not 32<=len(key)<=128:raise ValueError('Private operations key required')
    def auth(request:Request,response:Response):
        # Loopback proxy trust is configured by the host; token remains mandatory.
        value=request.headers.get('authorization','')
        if len(value)>160 or not hmac.compare_digest(value.encode(),('Bearer '+key).encode()):
            raise HTTPException(401,'web_admin_unauthorized')
        response.headers['Cache-Control']='no-store'
    router=APIRouter(prefix='/api/v1/_ops',dependencies=[Depends(auth)],include_in_schema=False)
    @router.get('/status')
    def status():return service.snapshot()
    @router.get('/requests')
    def requests(state:Literal['pending','approved','declined']='pending',before:UUID|None=None):
        return service.requests(state,str(before) if before else None)
    @router.post('/decision')
    def decision(body:Decision):return service.decision(body)
    @router.post('/quota')
    def quota(body:QuotaChange):return service.quota(body)
    return router
