"""Local-only fixed BigVGAN component endpoints, no client filesystem paths."""
from fastapi import APIRouter, Depends, HTTPException


def component_router(mode, identity, mutation):
    router = APIRouter()
    @router.get('/spec2wav/vocoder', operation_id='local_vocoder_status')
    def status(owner=Depends(identity)):
        if mode != 'local':
            raise HTTPException(404, 'unavailable')
        from ptb_worker.vocoder_components import status
        return status()
    @router.post('/spec2wav/vocoder/download', operation_id='download_local_vocoder')
    def download(owner=Depends(mutation)):
        if mode != 'local':
            raise HTTPException(404, 'unavailable')
        from ptb_worker.vocoder_components import install
        try:
            return install()
        except Exception as exc:
            code = str(exc)
            raise HTTPException(422, code if code.startswith(('m09_vocoder_', 'm11_')) and len(code)<80 else 'm09_vocoder_download_failed') from None
    return router
