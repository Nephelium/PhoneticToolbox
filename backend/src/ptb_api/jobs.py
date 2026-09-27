"""Every job, event and cancellation is scoped to the authenticated owner."""
import hmac
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, Request,Response
from .job_models import JobInput, FileJobInput, JobView, JobList, JobEvents, RetryInput
from ptb_worker.store import JobError
from .acoustic_batch_models import BatchRequest,BatchView,BatchList
from .spec2wav_models import Spec2WavRequest
from .egg_models import EggRequest
from .lpc_models import LpcRequest
from .m08_models import M08Request, M08Source, M08Manage
from .m14_models import M14Request
from .m06_models import M06Request
from .m07_models import M07Request
from .m11_models import M11Request
from .m05_models import M05Request, M05Upload, M05Block, M05Config
from .font_models import FigureFontSnapshot,FontPreflight


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

    from .m11_components import component_router
    router.include_router(component_router(ctx.mode,mutation))

    @router.get('/m05/catalog', operation_id='lip_catalog')
    def m05_catalog(owner=Depends(identity)):
        from ptb_worker.m05_task import catalog
        return catalog(ctx.mode=='local')

    @router.post('/m05/create', response_model=JobView, status_code=201, operation_id='create_lip_job')
    def m05_create(body:M05Request, owner=Depends(mutation)):
        from ptb_worker.m05_task import submit
        return submit(store,owner['id'],body)

    @router.post('/m05/{key}/repeat', response_model=JobView, status_code=201, operation_id='render_lip_animation')
    def m05_repeat(key:UUID, body:M05Config, owner=Depends(mutation)):
        from ptb_worker.m05_task import repeat
        return repeat(store,owner['id'],str(key),body)

    def m05_local():
        if ctx.mode!='local' or store.files is None:raise HTTPException(409,'m05_local_only')
        return store.files

    @router.post('/m05/uploads', operation_id='begin_local_lip_video')
    def m05_begin(body:M05Upload, owner=Depends(mutation)):
        from ptb_worker.m05_inputs import begin
        return begin(m05_local(),body.name,body.size)

    @router.put('/m05/uploads/{key}', operation_id='write_local_lip_video')
    def m05_block(key:UUID, body:M05Block, owner=Depends(mutation)):
        from ptb_worker.m05_inputs import block
        return block(m05_local(),str(key),body.offset,body.base64)

    @router.post('/m05/uploads/{key}/finalize', operation_id='finalize_local_lip_video')
    def m05_finish(key:UUID, owner=Depends(mutation)):
        from ptb_worker.m05_inputs import finish
        return finish(m05_local(),str(key))

    @router.post('/m05/uploads/{key}/abort', operation_id='abort_local_lip_video')
    def m05_abort(key:UUID, owner=Depends(mutation)):
        from ptb_worker.m05_inputs import abort
        return abort(m05_local(),str(key))

    @router.get('/m11/catalog',operation_id='mfa_component_catalog')
    def m11_catalog(owner=Depends(identity)):
        from ptb_worker.m11_task import catalog
        return catalog(local=ctx.mode=='local')

    @router.post('/m11/create',response_model=JobView,status_code=201,operation_id='create_mfa_alignment_job')
    def m11_create(body:M11Request,owner=Depends(mutation)):
        from ptb_worker.m11_task import submit
        return submit(store,owner['id'],body)

    @router.post('/m07/create',response_model=JobView,status_code=201,operation_id='create_phonation_synthesis_job')
    def create_m07(body:M07Request,owner=Depends(mutation)):
        from ptb_worker.m07_task import submit
        return submit(store,owner['id'],body)

    @router.post('/m06/create',response_model=JobView,status_code=201,operation_id='create_speech_synthesis_job')
    def create_m06(body:M06Request,owner=Depends(mutation)):
        from ptb_worker.m06_task import submit
        return submit(store,owner['id'],body)

    @router.post('/m14/create',response_model=JobView,status_code=201,operation_id='create_phonology_job')
    def create_m14(body:M14Request,owner=Depends(mutation)):
        from ptb_worker.m14_task import submit
        return submit(store,owner['id'],body)

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

    @router.post('/m08/create',response_model=JobView,status_code=201,operation_id='create_m08_job')
    def m08(body:M08Request,owner=Depends(mutation)):
        from ptb_worker.m08_task import submit
        return submit(store,owner['id'],body)

    @router.get('/m08/list/{project_id}',operation_id='list_m08_results')
    def m08_list(project_id:UUID,owner=Depends(identity)):
        from ptb_worker.m08_results import listing
        return listing(store,owner['id'],str(project_id))

    @router.post('/m08/history',operation_id='m08_history')
    def m08_history(body:M08Source,owner=Depends(mutation)):
        from ptb_worker.m08_results import listing
        return [r for j in listing(store,owner['id'],body.project_id,body.source.model_dump()) for r in j['results'] if r['saved']]

    @router.post('/m08/save',response_model=JobView,operation_id='save_m08_result')
    def m08_save(body:M08Manage,owner=Depends(mutation)):
        from ptb_worker.m08_results import save_copy
        return save_copy(store,owner['id'],body)

    @router.post('/m08/remove',operation_id='remove_m08_results')
    def m08_remove(body:M08Manage,owner=Depends(mutation)):
        from ptb_worker.m08_results import manage
        return manage(store,owner['id'],body,'remove')

    @router.post('/m08/rename',operation_id='rename_m08_results')
    def m08_rename(body:M08Manage,owner=Depends(mutation)):
        from ptb_worker.m08_results import manage
        return manage(store,owner['id'],body,'rename')

    @router.post('/lpc/create',response_model=JobView,status_code=201,operation_id='create_lpc_job')
    def lpc(body:LpcRequest,owner=Depends(mutation)):
        from ptb_worker.lpc_jobs import submit
        return submit(store,owner['id'],body)

    @router.post('/lpc/fonts',response_model=FontPreflight,operation_id='check_lpc_export_fonts')
    def lpc_fonts(body:FigureFontSnapshot,owner=Depends(mutation)):
        from ptb_worker.font_preflight import inspect_fonts
        return inspect_fonts(body)

    @router.post('/egg/create',response_model=JobView,status_code=201,operation_id='create_egg_job')
    def egg(body:EggRequest,owner=Depends(mutation)):
        from ptb_worker.egg_jobs import submit
        return submit(store,owner['id'],body)

    @router.post('/egg/fonts',response_model=FontPreflight,operation_id='check_egg_export_fonts')
    def egg_fonts(body:FigureFontSnapshot,owner=Depends(mutation)):
        from ptb_worker.font_preflight import inspect_fonts
        return inspect_fonts(body)

    @router.post('/local-inputs',operation_id='register_local_acoustic_input')
    async def local_input(request:Request,role:str,name:str,owner=Depends(mutation)):
        if ctx.mode!='local' or not getattr(store,'batches',None):raise HTTPException(404,'unavailable')
        limit=64_000_000 if role=='audio' else 16_000_000 if role=='dictionary' else 16_000_000 if role in ('parent_result','legacy_result','image') else 2_000_000
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
