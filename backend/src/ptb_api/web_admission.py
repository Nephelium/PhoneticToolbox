"""Bounded HTTP work with reserved recovery capacity, enabled only for web workspaces."""
import asyncio
from starlette.responses import JSONResponse


class Lane:
    def __init__(self, slots, waiting):
        self.semaphore=asyncio.Semaphore(slots)
        self.limit=slots+waiting
        self.entered=0


class WebAdmission:
    def __init__(self, app, *, slots=8, recovery_slots=4, waiting=24, timeout=1.5):
        self.app=app;self.timeout=timeout
        self.normal=Lane(slots,waiting);self.recovery=Lane(recovery_slots,16)

    async def __call__(self,scope,receive,send):
        path=scope.get('path','');method=scope.get('method','')
        if scope['type']!='http' or not path.startswith('/api/v1/') or path=='/api/v1/health':
            return await self.app(scope,receive,send)
        recovery=(method=='POST' and (path in ('/api/v1/workspaces/heartbeat','/api/v1/auth/logout')
            or path.startswith('/api/v1/jobs/') and path.endswith('/cancel'))
            or method=='DELETE' and path.startswith(('/api/v1/assets/','/api/v1/vocal/sessions/')))
        lane=self.recovery if recovery else self.normal
        async def reject():
            await JSONResponse({'detail':'web_server_busy'},status_code=429,
                headers={'Retry-After':'5','Cache-Control':'no-store'})(scope,receive,send)
        # ASGI calls share one event loop; no await between checking and counting.
        if lane.entered>=lane.limit:return await reject()
        lane.entered+=1;acquired=False
        try:
            try:
                await asyncio.wait_for(lane.semaphore.acquire(),self.timeout);acquired=True
            except TimeoutError:return await reject()
            await self.app(scope,receive,send)
        finally:
            if acquired:lane.semaphore.release()
            lane.entered-=1
