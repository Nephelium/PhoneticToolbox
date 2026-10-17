import asyncio
import httpx
from ptb_api.web_admission import WebAdmission


def test_recovery_lane_survives_full_normal_lane_and_releases_on_failure():
    async def scenario():
        entered=asyncio.Event();release=asyncio.Event()
        async def app(scope,receive,send):
            if scope['path'].endswith('/projects'):
                entered.set();await release.wait()
            if scope['path'].endswith('/fail'):raise RuntimeError('synthetic')
            from starlette.responses import JSONResponse
            await JSONResponse({'ok':True})(scope,receive,send)
        gate=WebAdmission(app,slots=1,recovery_slots=1,waiting=0,timeout=.02)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gate),base_url='http://test') as c:
            work=asyncio.create_task(c.get('/api/v1/projects'));await entered.wait()
            busy=await c.get('/api/v1/projects')
            assert busy.status_code==429 and busy.headers['retry-after']=='5'
            for path in ('/api/v1/workspaces/heartbeat','/api/v1/auth/logout','/api/v1/jobs/a/cancel'):
                assert (await c.post(path)).status_code==200
            assert (await c.delete('/api/v1/assets/a')).status_code==200
            assert (await c.get('/api/v1/health')).status_code==200
            release.set();assert (await work).status_code==200
            try:await c.get('/api/v1/fail')
            except RuntimeError:pass
            assert (await c.get('/api/v1/projects')).status_code==200
        assert gate.normal.entered==gate.recovery.entered==0
    asyncio.run(scenario())


def test_waiting_timeout_and_cancel_do_not_leak_slots():
    async def scenario():
        entered=asyncio.Event();release=asyncio.Event()
        async def app(scope,receive,send):entered.set();await release.wait()
        gate=WebAdmission(app,slots=1,waiting=1,timeout=.02)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gate),base_url='http://test') as c:
            active=asyncio.create_task(c.get('/api/v1/projects'));await entered.wait()
            assert (await c.get('/api/v1/projects')).status_code==429
            pending=asyncio.create_task(c.get('/api/v1/projects'));await asyncio.sleep(0)
            pending.cancel();active.cancel()
            await asyncio.gather(pending,active,return_exceptions=True)
            assert gate.normal.entered==0 and gate.normal.semaphore._value==1
    asyncio.run(scenario())
