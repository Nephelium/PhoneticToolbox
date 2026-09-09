"""Every job, event and cancellation is scoped to the authenticated owner."""
import hmac
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from .job_models import JobInput, FileJobInput, JobView, JobList, JobEvents, RetryInput
from ptb_worker.store import JobError


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
    return router
