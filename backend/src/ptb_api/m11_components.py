"""Local host-only optional component operations; never exposed to web accounts."""
from pathlib import Path
import json
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field
from .models import WireModel


class LocalM11ComponentRequest(WireModel):
    action: Literal['check','import']
    runtime: str | None = Field(default=None,max_length=4096)
    model: str = Field(max_length=4096)
    dictionary: str = Field(max_length=4096)
    archive: str | None = Field(default=None,max_length=4096)
    manifest: str | None = Field(default=None,max_length=4096)
    trusted_manifest_sha256: str | None = Field(default=None,pattern=r'^[a-f0-9]{64}$')


def component_router(mode,mutation):
    router=APIRouter()

    @router.post('/m11/component',operation_id='manage_local_mfa_component')
    def component(body:LocalM11ComponentRequest,owner=Depends(mutation)):
        if mode!='local':raise HTTPException(404,'unavailable')
        from ptb_worker.mfa.probe import register,publish_registration
        from ptb_worker.mfa.components import ComponentManager,digest,no_links
        from ptb_worker.mfa.runtime import registry_root
        try:
            if body.action=='check':
                if not body.runtime:raise ValueError('m11_runtime_missing')
                result=register(body.runtime,body.model,body.dictionary)
                return {k:result[k] for k in ('runtime_id','model_id','receipt')}
            if not all((body.archive,body.manifest,body.trusted_manifest_sha256)):
                raise ValueError('m11_manifest_trust_required')
            manifest=Path(body.manifest);no_links(manifest)
            if manifest.stat().st_size>32_000_000 or digest(manifest)!=body.trusted_manifest_sha256:
                raise ValueError('m11_manifest_hash_mismatch')
            trusted=json.loads(manifest.read_text(encoding='utf8'))
            manager=ComponentManager(registry_root())
            prepared={}
            def check(target):
                # No arbitrary install hooks. A relocatable publisher must supply
                # a runtime that works at its final path, demonstrated by this task.
                prepared.update(register(target,body.model,body.dictionary,publish=False))
                return dict(success=True,receipt=prepared['receipt'])
            manager.import_archive(body.archive,trusted,check)
            prepared['runtime_record'].update(download_bytes=trusted['download_bytes'],installed_bytes=trusted['installed_bytes'],source=trusted['source'],archive_sha256=trusted['sha256'])
            publish_registration(prepared,registry_root())
            return dict(success=True,runtime_id=prepared['runtime_id'],model_id=prepared['model_id'])
        except (ValueError,OSError,KeyError) as exc:
            code=str(exc)
            raise HTTPException(422,code if code.startswith('m11_') and len(code)<80 else 'm11_component_failed') from None
    return router
