"""Owner-scoped P07 endpoints. Raw upload blocks are bounded by AccountBoundary."""
from typing import Literal
from urllib.parse import quote
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response
from .quota import CHUNK_BYTES, StorageError, content_range
from .storage_models import UploadInput, FinalizeInput, AssetView, AssetList, StorageUsage, DeleteImpact


class PrivateDownload(Response):
    def __init__(self, store, ctx, request, owner, asset, start, end, partial):
        headers = {'Content-Disposition': "attachment; filename*=UTF-8''" + quote(asset['name'], safe=''),
                   'Content-Length': str(end-start), 'Accept-Ranges': 'bytes',
                   'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'}
        if partial:
            headers['Content-Range'] = f"bytes {start}-{end-1}/{asset['size_bytes']}"
        super().__init__(b'', status_code=206 if partial else 200, headers=headers,
                         media_type='application/octet-stream')
        self.store, self.ctx, self.request = store, ctx, request
        self.owner, self.asset, self.start, self.end = owner, asset, start, end

    async def __call__(self, scope, receive, send):
        await send({'type': 'http.response.start', 'status': self.status_code, 'headers': self.raw_headers})
        offset = self.start
        while offset < self.end:
            # Never retain an open file across network back-pressure or expiry.
            session = await run_in_threadpool(self.ctx.session, self.request)
            if session['id'] != self.owner:
                raise StorageError('account_changed', 409)
            block = await run_in_threadpool(self.store.read_block, self.owner, self.asset['id'],
                                            offset, min(CHUNK_BYTES, self.end-offset))
            if not block:
                raise StorageError('storage_read_failed', 503)
            await send({'type': 'http.response.body', 'body': block, 'more_body': True})
            offset += len(block)
        await send({'type': 'http.response.body', 'body': b'', 'more_body': False})


def create_storage_router(ctx, store):
    router = APIRouter(prefix='/api/v1', tags=['storage'])

    def available():
        ctx.available()
        if store is None:
            raise HTTPException(503, 'storage_service_unavailable')

    def identity(request: Request):
        available()
        return ctx.session(request)

    def mutation(request: Request):
        available()
        return ctx.mutation(request)

    @router.get('/storage/usage', response_model=StorageUsage, operation_id='get_storage_usage')
    def usage(owner=Depends(identity)):
        return store.usage(owner['id'])

    @router.get('/assets', response_model=AssetList, operation_id='list_assets')
    def listing(project_id: UUID, order: Literal['expires', 'size', 'created'] = 'expires', owner=Depends(identity)):
        return {'assets': store.list(owner['id'], project_id, order)}

    @router.post('/uploads', response_model=AssetView, status_code=201, operation_id='create_upload')
    def create(body: UploadInput, owner=Depends(mutation)):
        return store.create(owner['id'], body)

    @router.put('/uploads/{asset_id}/blocks', response_model=AssetView, operation_id='append_upload_block',
                openapi_extra={'requestBody': {'required': True, 'content': {'application/octet-stream': {'schema': {'type': 'string', 'format': 'binary'}}}}})
    async def append(asset_id: UUID, request: Request, offset: int = Query(ge=0), owner=Depends(mutation)):
        if request.headers.get('content-type') != 'application/octet-stream':
            raise HTTPException(415, 'binary_chunk_required')
        data = await request.body()
        return await run_in_threadpool(store.append, owner['id'], asset_id, offset, data)

    @router.post('/uploads/{asset_id}/finalize', response_model=AssetView, operation_id='finalize_upload')
    def finalize(asset_id: UUID, body: FinalizeInput, owner=Depends(mutation)):
        return store.finalize(owner['id'], asset_id, body.sha256)

    @router.get('/assets/{asset_id}', response_model=AssetView, operation_id='get_asset')
    def metadata(asset_id: UUID, owner=Depends(identity)):
        return store.metadata(owner['id'], asset_id)

    @router.get('/assets/{asset_id}/content', response_class=Response, operation_id='download_asset',
                responses={200: {'content': {'application/octet-stream': {}}}, 206: {'description': 'Partial content'},
                           416: {'description': 'Invalid or unsatisfiable single range'}})
    def download(asset_id: UUID, request: Request, expected_account: UUID | None = None, owner=Depends(identity)):
        if expected_account is not None and str(expected_account) != owner['id']:
            raise HTTPException(409, 'account_changed')
        asset = store.metadata(owner['id'], asset_id)
        try:
            start, end, partial = content_range(request.headers.get('range'), asset['size_bytes'])
        except StorageError:
            return Response(status_code=416, headers={'Content-Range': f"bytes */{asset['size_bytes']}", 'Cache-Control': 'no-store'})
        return PrivateDownload(store, ctx, request, owner['id'], asset, start, end, partial)

    @router.delete('/assets/{asset_id}', response_model=AssetView, operation_id='delete_asset')
    def delete(asset_id: UUID, owner=Depends(mutation)):
        return store.delete(owner['id'], asset_id)

    @router.get('/assets/{asset_id}/delete-impact', response_model=DeleteImpact, operation_id='get_delete_impact')
    def impact(asset_id: UUID, owner=Depends(identity)):
        return store.impact(owner['id'],asset_id)

    return router
