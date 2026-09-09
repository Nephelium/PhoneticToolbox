"""Local authenticated raw audio preview; server uses existing owned asset IDs."""
import hashlib
import hmac
from fastapi import APIRouter,HTTPException,Request,Query
from starlette.concurrency import run_in_threadpool
from ptb_worker.spectrogram_preview import render,preview_slot,MAX_BYTES
from .preview_models import SpectrogramPreview


def create_preview_router(mode,token,origin):
    router=APIRouter(prefix='/api/v1/preview',tags=['preview'])

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
