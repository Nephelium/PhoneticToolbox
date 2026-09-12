"""Every job, event and cancellation is scoped to the authenticated owner."""
import hmac
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, Request,Response
from .job_models import JobInput, FileJobInput, JobView, JobList, JobEvents, RetryInput
from ptb_worker.store import JobError
from .acoustic_batch_models import BatchRequest,BatchView,BatchList
from .spec2wav_models import Spec2WavRequest


def create_job_router(ctx, store, *, local_token=None, local_origin=None):
    router=APIRouter(prefix='/api/v1/jobs',tags=['jobs'])
    def identity(request: Request):
        if store is None:raise HTTPException(503,'task_service_unavailable')
        if ctx.mode=='server':return ctx.session(request)
        expected='Bearer '+local_token if local_token else ''
        if not expected or not hmac.compare_digest(request.headers.get('authorization','').encode(),expected.encode()):
            raise HTTPException(403,'permission_denied')
        return {'id':'local'}

    def mutation(request: Request):
        if store is None:raise HTTPException(503,'task_service_unavailable')
        if ctx.mode=='server':return ctx.mutation(request)
        owner=identity(request)
        if not local_origin or request.headers.getlist('origin') != [local_origin]:
            raise HTTPException(403,'origin_rejected')
        return owner

    @router.post('',response_model=JobView,status_code=201,operation_id='create_job')
    def create(body: JobInput | FileJobInput, owner=Depends(mutation)):
        return store.submit(owner['id'],body)

    @router.get('',response_model=JobList,operation_id='list_jobs')
    def listing(project_id: UUID, owner=Depends(identity)):
        return {'jobs':store.list(owner['id'],str(project_id))}

    @router.get('/{job_id}',response_model=JobView,operation_id='get_job')
    def get(job_id: UUID,owner=Depends(identity)):
        return store.get(owner['id'],str(job_id))

    @router.get('/{job_id}/events',response_model=JobEvents,operation_id='get_job_events')
    def events(job_id: UUID,after: int=Query(0,ge=0,le=1000000),owner=Depends(identity)):
        return {'events':store.events(owner['id'],str(job_id),after)}

    @router.post('/{job_id}/cancel',response_model=JobView,operation_id='cancel_job')
    def cancel(job_id: UUID,owner=Depends(mutation)):
        return store.cancel(owner['id'],str(job_id))

    @router.post('/{job_id}/retry',response_model=JobView,status_code=201,operation_id='retry_job')
    def retry(job_id: UUID,body: RetryInput,owner=Depends(mutation)):
        return store.retry(owner['id'],str(job_id),body.idempotency_key)

    def batches():
        result=getattr(store,'batches',None)
        if result is None:raise HTTPException(503,'acoustic_tasks_unavailable')
        return result

    @router.post('/batches/create',response_model=BatchView,status_code=201,operation_id='create_acoustic_batch')
    def create_batch(body:BatchRequest,owner=Depends(mutation)):
        return batches().submit(owner['id'],body)

    @router.get('/batches/list',response_model=BatchList,operation_id='list_acoustic_batches')
    def list_batches(project_id:UUID,owner=Depends(identity)):
        return {'batches':batches().list(owner['id'],str(project_id))}

    @router.get('/batches/{batch_id}',response_model=BatchView,operation_id='get_acoustic_batch')
    def get_batch(batch_id:UUID,owner=Depends(identity)):
        return batches().get(owner['id'],str(batch_id))

    @router.post('/batches/{batch_id}/cancel',response_model=BatchView,operation_id='cancel_acoustic_batch')
    def cancel_batch(batch_id:UUID,owner=Depends(mutation)):
        return batches().cancel(owner['id'],str(batch_id))

    @router.post('/spec2wav/create',response_model=JobView,status_code=201,operation_id='create_spec2wav_job')
    def reconstruct(body:Spec2WavRequest,owner=Depends(mutation)):
        from ptb_worker.spec2wav_jobs import submit
        return submit(store,owner['id'],body)

    @router.post('/local-inputs',operation_id='register_local_acoustic_input')
    async def local_input(request:Request,role:str,name:str,owner=Depends(mutation)):
        if ctx.mode!='local' or not getattr(store,'batches',None):raise HTTPException(404,'unavailable')
        limit=64_000_000 if role=='audio' else 16_000_000 if role in ('parent_result','legacy_result','image') else 2_000_000
        raw=bytearray()
        async for block in request.stream():
            if len(raw)+len(block)>limit:raise HTTPException(413,'input_budget_exceeded')
            raw.extend(block)
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(store.files.import_input,bytes(raw),name,role)

    @router.get('/parents/latest',operation_id='find_acoustic_parent')
    def parent(project_id:UUID,sha256:str=Query(pattern=r'^[0-9a-f]{64}$'),owner=Depends(identity)):
        return batches().parent_result(owner['id'],str(project_id),sha256)

    @router.post('/local-lip-conversion',operation_id='convert_local_legacy_lip')
    async def convert_lip(request:Request,owner=Depends(mutation)):
        if ctx.mode!='local' or not getattr(store,'batches',None):raise HTTPException(404,'unavailable')
        from ptb_worker.legacy_conversion import convert,MAX_INPUT
        from ptb_worker.spectrogram_preview import preview_slot
        from starlette.concurrency import run_in_threadpool
        with preview_slot():
            raw=bytearray()
            async for block in request.stream():
                if len(raw)+len(block)>MAX_INPUT:raise HTTPException(413,'legacy_conversion_budget')
                raw.extend(block)
            result=await run_in_threadpool(convert,bytes(raw))
        return Response(result,media_type='application/json')

    @router.get('/local-results/{asset_id}',operation_id='read_local_acoustic_result')
    def local_result(asset_id:UUID,offset:int=Query(0,ge=0),size:int=Query(65536,ge=1,le=1048576),owner=Depends(identity)):
        if ctx.mode!='local' or not getattr(store,'batches',None):raise HTTPException(404,'unavailable')
        return Response(store.files.read_result(owner['id'],str(asset_id),offset,size),media_type='application/octet-stream')
    return router
