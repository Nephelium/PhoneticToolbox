"""Public P07 metadata; disk keys, leases and account identity stay on the server."""
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .quota import QUOTA_BYTES


class UploadInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    project_id: UUID
    name: str = Field(min_length=1, max_length=180)
    expected_bytes: int | None = Field(default=None, ge=0, le=QUOTA_BYTES, strict=True)
    idempotency_key: str = Field(min_length=16, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')

    @field_validator('name')
    @classmethod
    def display_name(cls, value):
        if not value.strip() or any(ord(c) < 32 or c in '/\\:' for c in value):
            raise ValueError('Invalid display filename')
        return value.strip()


class FinalizeInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sha256: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')


class AssetView(BaseModel):
    id: str
    project_id: str
    name: str
    kind: Literal['input']
    state: Literal['uploading', 'ready', 'deleting', 'delete_failed', 'deleted']
    size_bytes: int
    reserved_bytes: int
    expected_bytes: int | None
    sha256: str | None
    created_at: float
    expires_at: float
    error_code: str | None


class AssetList(BaseModel):
    assets: list[AssetView]


class StorageUsage(BaseModel):
    quota_bytes: int
    used_bytes: int
    reserved_bytes: int
    available_bytes: int
    frozen: bool
    ready: bool
