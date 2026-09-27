"""Unmounted remote router. Mount only after A's transaction bridge is verified."""
import hashlib
import asyncio
from uuid import UUID
from fastapi import APIRouter, Request, HTTPException, Query, Response
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool
from .remote_models import Poll, Health, Heartbeat, Output, AttemptRequest, Failure
from .remote_models import Claim, LeaseResponse, HealthResponse, UploadResponse, OffsetResponse, CompleteResponse, FailResponse
from ptb_worker.remote_scheduler import CHUNK_BYTES
from ptb_worker.store import JobError


class BoundedRemoteRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def bounded(request):
            if request.method == 'POST':
                data = bytearray()
                try:
                    async with asyncio.timeout(10):
                        async for part in request.stream():
                            if len(data)+len(part) > 16384:
                                raise HTTPException(413, detail={'code': 'invalid_request'})
                            data.extend(part)
                except TimeoutError:
                    raise HTTPException(408, detail={'code': 'transfer_stalled'}) from None
                request._body = bytes(data)
            response = await original(request)
            response.headers['Cache-Control'] = 'no-store'
            return response
        return bounded


def create_remote_router(coordinator):
    router = APIRouter(prefix='/api/v1/worker', tags=['remote/1'], route_class=BoundedRemoteRoute)

    def token(request):
        # Do not trust arbitrary X-Forwarded-Proto here; trusted proxy setup is deployment work.
        auth = request.headers.get('Authorization', '')
        if request.url.scheme != 'https' or request.query_params.get('token') or not auth.startswith('Bearer '):
            raise HTTPException(401, detail={'code': 'node_unauthorized'})
        return auth[7:]

    def invoke(call, *args):
        try: return call(*args)
        except JobError as error: raise HTTPException(error.status, detail={'code': error.code}) from None

    @router.post('/health', response_model=HealthResponse)
    def health(request: Request, body: Health):
        return invoke(coordinator.health, token(request), body.runtime_hash)

    @router.post('/poll', response_model=Claim | None)
    def poll(request: Request, body: Poll):
        return invoke(coordinator.claim, token(request), body.request_id, body.runtime_hash)

    @router.post('/attempts/{aid}/heartbeat', response_model=LeaseResponse)
    def heartbeat(request: Request, aid: UUID, body: Heartbeat):
        return invoke(coordinator.heartbeat, token(request), str(aid), body.generation, body.phase, body.node_bytes)

    @router.get('/attempts/{aid}/inputs/{asset_id}')
    def read(request: Request, aid: UUID, asset_id: UUID, generation: int = Query(ge=1),
             offset: int = Query(default=0, ge=0), size: int = Query(default=65536, ge=1, le=CHUNK_BYTES)):
        data = invoke(coordinator.read, token(request), str(aid), generation, str(asset_id), offset, size)
        return Response(data, media_type='application/octet-stream', headers={
            'Cache-Control': 'no-store', 'X-Chunk-SHA256': hashlib.sha256(data).hexdigest()})

    @router.post('/attempts/{aid}/outputs', response_model=UploadResponse)
    def output(request: Request, aid: UUID, body: Output):
        return invoke(coordinator.output, token(request), str(aid), body)

    @router.put('/attempts/{aid}/outputs/{uid}', response_model=OffsetResponse)
    async def write(request: Request, aid: UUID, uid: UUID, generation: int = Query(ge=1), offset: int = Query(ge=0)):
        credential = token(request)
        # Authenticate/fence BEFORE accepting a potentially slow request body.
        await run_in_threadpool(invoke, coordinator.check, credential, str(aid), generation)
        data = bytearray()
        try:
            async with asyncio.timeout(10):
                async for part in request.stream():
                    if len(data)+len(part) > CHUNK_BYTES:
                        raise HTTPException(413, detail={'code': 'invalid_chunk'})
                    data.extend(part)
        except TimeoutError:
            raise HTTPException(408, detail={'code': 'transfer_stalled'}) from None
        return await run_in_threadpool(invoke, coordinator.write, credential, str(aid), generation, str(uid), offset,
                                       bytes(data), request.headers.get('X-Chunk-SHA256', ''))

    @router.post('/attempts/{aid}/complete', response_model=CompleteResponse)
    def complete(request: Request, aid: UUID, body: AttemptRequest):
        return {'result_manifest': invoke(coordinator.complete, token(request), str(aid), body.generation)}

    @router.post('/attempts/{aid}/fail', response_model=FailResponse)
    def fail(request: Request, aid: UUID, body: Failure):
        return invoke(coordinator.fail, token(request), str(aid), body.generation, body.code)

    return router
