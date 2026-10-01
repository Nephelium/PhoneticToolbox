"""Local authenticated raw audio preview; server uses existing owned asset IDs."""
import hashlib
import hmac
from fastapi import APIRouter,HTTPException,Request,Query
from starlette.concurrency import run_in_threadpool
from ptb_worker.spectrogram_preview import render,preview_slot,MAX_BYTES
from .preview_models import SpectrogramPreview
from .display_models import ParameterTable
from .egg_models import EggTaskConfig
from .egg_interactive_models import EggPreviewSession, EggInteractiveResult
from uuid import UUID


def create_preview_router(mode,token,origin,*,egg_preview=None):
    router=APIRouter(prefix='/api/v1/preview',tags=['preview'])

    def egg_auth(request):
        if mode != 'local' or not token or not origin or egg_preview is None: raise HTTPException(404,'local_preview_unavailable')
        if request.headers.get('origin') != origin or not hmac.compare_digest(request.headers.get('authorization','').encode(), ('Bearer '+token).encode()): raise HTTPException(403,'permission_denied')

    @router.post('/egg', response_model=EggPreviewSession, operation_id='open_local_egg_preview')
    async def egg_open(request:Request):
        egg_auth(request)
        if request.headers.get('content-type') != 'application/octet-stream': raise HTTPException(415,'binary_audio_required')
        data = bytearray()
        async for chunk in request.stream():
            if len(data)+len(chunk) > MAX_BYTES: raise HTTPException(413,'egg_input_budget')
            data.extend(chunk)
        return await run_in_threadpool(egg_preview.open, 'local', bytes(data))

    @router.post('/egg/{session_id}', response_model=EggInteractiveResult, operation_id='update_local_egg_preview')
    async def egg_update(session_id:UUID, body:EggTaskConfig, request:Request):
        egg_auth(request)
        import json
        value=await run_in_threadpool(egg_preview.update, 'local', session_id, body.model_dump())
        return EggInteractiveResult.model_validate_json(json.dumps(value))

    @router.delete('/egg/{session_id}', operation_id='close_local_egg_preview')
    async def egg_close(session_id:UUID, request:Request):
        egg_auth(request)
        await run_in_threadpool(egg_preview.release, 'local', session_id)
        return {'closed':True}

    @router.post('/parameters',response_model=ParameterTable,operation_id='local_parameter_table')
    async def parameters(request:Request,name:str=Query(max_length=220)):
        if mode!='local' or not token or not origin:raise HTTPException(404,'local_preview_unavailable')
        if request.headers.get('origin')!=origin or not hmac.compare_digest(request.headers.get('authorization','').encode(),('Bearer '+token).encode()):
            raise HTTPException(403,'permission_denied')
        from ptb_worker.parameter_preview import render as table_render,MAX_BYTES as table_limit
        with preview_slot():
            data=bytearray()
            async for chunk in request.stream():
                if len(data)+len(chunk)>table_limit:raise HTTPException(413,'parameter_input_budget')
                data.extend(chunk)
            return await run_in_threadpool(table_render,bytes(data),name)

    @router.post('/spectrogram',response_model=SpectrogramPreview,operation_id='local_spectrogram_preview')
    async def spectrogram(request:Request,channel:int=Query(ge=0,le=31),start:float=Query(ge=0),
                          end:float=Query(gt=0),width:int=Query(default=800,ge=100,le=1000)):
        if mode!='local' or not token or not origin:raise HTTPException(404,'local_preview_unavailable')
        if request.headers.get('origin')!=origin or not hmac.compare_digest(request.headers.get('authorization','').encode(),('Bearer '+token).encode()):
            raise HTTPException(403,'permission_denied')
        if request.headers.get('content-type')!='application/octet-stream':raise HTTPException(415,'binary_audio_required')
        with preview_slot():
            data=bytearray()
            async for chunk in request.stream():
                if len(data)+len(chunk)>MAX_BYTES:raise HTTPException(413,'preview_too_large')
                data.extend(chunk)
            raw=bytes(data);del data
            value=await run_in_threadpool(render,raw,channel=channel,start=start,end=end,width=width)
        return dict(sha256=hashlib.sha256(raw).hexdigest(),**value)
    return router
