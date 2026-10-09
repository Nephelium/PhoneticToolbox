"""Authenticated local result-retention controls, separate from research data."""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class RetentionPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    enabled: bool
    days: int = Field(ge=1, le=3650)
    exported_cache_days: int = Field(default=7, ge=1, le=3650)


class ExportReceipt(BaseModel):
    model_config = ConfigDict(extra='forbid')
    assets: list[UUID] = Field(min_length=1, max_length=3004)


def retention_router(local, store, identity, mutation):
    router = APIRouter()
    def manager():
        if not local or store is None or store.files is None:
            raise HTTPException(409, 'local_retention_only')
        from ptb_worker.local_retention import LocalRetention
        return LocalRetention(store.files)

    @router.get('/local-storage', operation_id='local_storage_status')
    def status(owner=Depends(identity)):
        return manager().status()

    @router.post('/local-storage/policy', operation_id='configure_local_retention')
    def policy(body: RetentionPolicy, owner=Depends(mutation)):
        return manager().configure(**body.model_dump())

    @router.post('/local-storage/cleanup', operation_id='clean_expired_local_results')
    def cleanup(owner=Depends(mutation)):
        return manager().sweep(force=True)

    @router.post('/local-storage/clear-all', operation_id='clear_all_local_result_caches')
    def clear_all(owner=Depends(mutation)):
        return manager().sweep(force=True, all_cache=True)

    @router.post('/local-storage/export-receipt', operation_id='acknowledge_local_export')
    def exported(body: ExportReceipt, owner=Depends(mutation)):
        return manager().exported([str(key) for key in body.assets])
    return router
